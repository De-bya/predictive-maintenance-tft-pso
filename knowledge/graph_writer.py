"""
knowledge/graph_writer.py

Bridges the existing TFT prediction + explainability pipeline
(evaluation/explain.py) to a Neo4j knowledge graph.

Every time /predict returns a FAULT, call log_fault_event() with
the same top-3 sensor importance output extract_sensor_importance()
already computes — no new ML logic needed here, this module only
persists what the model already produced and lets you query history.

Requires: pip install neo4j
Env vars: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from neo4j import GraphDatabase, basic_auth


class GraphStore:
    def __init__(self, uri=None, user=None, password=None):
        self._uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.environ.get("NEO4J_USER", "neo4j")
        self._password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self._driver = GraphDatabase.driver(self._uri, auth=basic_auth(self._user, self._password))

    def close(self):
        self._driver.close()

    # ------------------------------------------------------------
    # WRITE PATH — called from api.py right after a FAULT prediction
    # ------------------------------------------------------------
    def log_fault_event(
        self,
        machine_id: str,
        confidence: float,
        top_sensors: list,          # [{"rank": 1, "sensor": "PS1", "importance": 0.91}, ...]
        source: str = "pytorch",    # "pytorch" | "onnx"
        drift_score: Optional[float] = None,
        model_version: str = "unknown",
    ) -> str:
        """Persist one fault prediction + its explainability output as a graph node.
        Returns the generated event_id."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._driver.session() as session:
            session.execute_write(
                self._write_fault_event_tx,
                event_id, machine_id, timestamp, confidence,
                top_sensors, source, drift_score, model_version,
            )
        return event_id

    @staticmethod
    def _write_fault_event_tx(tx, event_id, machine_id, timestamp, confidence,
                               top_sensors, source, drift_score, model_version):
        tx.run(
            """
            MERGE (m:Machine {machine_id: $machine_id})
            CREATE (f:FaultEvent {
                event_id: $event_id, timestamp: $timestamp, confidence: $confidence,
                status: "open", source: $source, drift_score: $drift_score,
                model_version: $model_version
            })
            MERGE (f)-[:OCCURRED_ON]->(m)
            """,
            machine_id=machine_id, event_id=event_id, timestamp=timestamp,
            confidence=confidence, source=source, drift_score=drift_score,
            model_version=model_version,
        )
        for s in top_sensors:
            tx.run(
                """
                MATCH (f:FaultEvent {event_id: $event_id})
                MATCH (s:Sensor {sensor_id: $sensor_id})
                MERGE (f)-[r:FLAGGED_BY]->(s)
                SET r.importance = $importance, r.rank = $rank
                """,
                event_id=event_id, sensor_id=s["sensor"],
                importance=s["importance"], rank=s["rank"],
            )

    def log_resolution(self, event_id: str, description: str, outcome: str,
                        technician_id: Optional[str] = None, part_ids: Optional[list] = None):
        """Attach a MaintenanceAction to a FaultEvent once it's been fixed
        — this is what builds up the 'what fixed similar faults before' history."""
        action_id = str(uuid.uuid4())
        performed_at = datetime.now(timezone.utc).isoformat()
        with self._driver.session() as session:
            session.execute_write(
                self._log_resolution_tx, event_id, action_id, description,
                outcome, performed_at, technician_id, part_ids or [],
            )
        return action_id

    @staticmethod
    def _log_resolution_tx(tx, event_id, action_id, description, outcome,
                            performed_at, technician_id, part_ids):
        tx.run(
            """
            MATCH (f:FaultEvent {event_id: $event_id})
            CREATE (a:MaintenanceAction {
                action_id: $action_id, description: $description,
                outcome: $outcome, performed_at: $performed_at
            })
            MERGE (f)-[:RESOLVED_BY]->(a)
            SET f.status = "resolved"
            """,
            event_id=event_id, action_id=action_id, description=description,
            outcome=outcome, performed_at=performed_at,
        )
        if technician_id:
            tx.run(
                """
                MATCH (a:MaintenanceAction {action_id: $action_id})
                MERGE (t:Technician {technician_id: $technician_id})
                MERGE (a)-[:PERFORMED_BY]->(t)
                """,
                action_id=action_id, technician_id=technician_id,
            )
        for part_id in part_ids:
            tx.run(
                """
                MATCH (a:MaintenanceAction {action_id: $action_id})
                MERGE (p:SparePart {part_id: $part_id})
                MERGE (a)-[:USED]->(p)
                """,
                action_id=action_id, part_id=part_id,
            )

    # ------------------------------------------------------------
    # READ PATH — this is what the LangChain graph_tool calls
    # ------------------------------------------------------------
    def find_similar_past_faults(self, top_sensor_ids: list, min_overlap: int = 2, limit: int = 5):
        """Given the top sensors implicated in a new prediction, find past
        FaultEvents that were flagged by an overlapping set of sensors,
        and return how each one was resolved (if it was)."""
        query = """
        MATCH (past:FaultEvent)-[:FLAGGED_BY]->(s:Sensor)
        WHERE s.sensor_id IN $sensor_ids
        WITH past, count(DISTINCT s) AS overlap
        WHERE overlap >= $min_overlap
        OPTIONAL MATCH (past)-[:RESOLVED_BY]->(action:MaintenanceAction)
        RETURN past.event_id AS event_id, past.timestamp AS timestamp,
               past.status AS status, overlap AS sensor_overlap,
               action.description AS resolution, action.outcome AS outcome
        ORDER BY overlap DESC, past.timestamp DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            result = session.run(query, sensor_ids=top_sensor_ids, min_overlap=min_overlap, limit=limit)
            return [dict(r) for r in result]

    def machine_fault_history(self, machine_id: str, limit: int = 10):
        query = """
        MATCH (f:FaultEvent)-[:OCCURRED_ON]->(m:Machine {machine_id: $machine_id})
        OPTIONAL MATCH (f)-[:FLAGGED_BY]->(s:Sensor)
        WITH f, collect(s.sensor_id) AS sensors
        RETURN f.event_id AS event_id, f.timestamp AS timestamp,
               f.status AS status, f.confidence AS confidence, sensors
        ORDER BY f.timestamp DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            result = session.run(query, machine_id=machine_id, limit=limit)
            return [dict(r) for r in result]


# ------------------------------------------------------------
# Convenience singleton so api.py / agent code can just import
# and call get_graph_store() without wiring connection params
# everywhere.
# ------------------------------------------------------------
_graph_store_instance = None

def get_graph_store() -> GraphStore:
    global _graph_store_instance
    if _graph_store_instance is None:
        _graph_store_instance = GraphStore()
    return _graph_store_instance


if __name__ == "__main__":
    # Quick manual smoke test — requires a running Neo4j instance
    gs = get_graph_store()
    event_id = gs.log_fault_event(
        machine_id="HYD-01",
        confidence=0.94,
        top_sensors=[
            {"rank": 1, "sensor": "PS1", "importance": 0.91},
            {"rank": 2, "sensor": "TS2", "importance": 0.63},
            {"rank": 3, "sensor": "FS1", "importance": 0.41},
        ],
        model_version="tft-v1",
    )
    print("Logged fault event:", event_id)
    print("Similar past faults:", gs.find_similar_past_faults(["PS1", "TS2"]))
    gs.close()
