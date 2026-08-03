// ============================================================
// Predictive Maintenance Knowledge Graph -- Schema
// Run this once against a fresh Neo4j database (neo4j-cypher-shell
// or the Neo4j Browser) before running graph_writer.py for the
// first time.
// ============================================================

// ---------- Constraints (also create the backing indexes) ----------
CREATE CONSTRAINT machine_id IF NOT EXISTS
FOR (m:Machine) REQUIRE m.machine_id IS UNIQUE;

CREATE CONSTRAINT sensor_id IF NOT EXISTS
FOR (s:Sensor) REQUIRE s.sensor_id IS UNIQUE;

CREATE CONSTRAINT fault_event_id IF NOT EXISTS
FOR (f:FaultEvent) REQUIRE f.event_id IS UNIQUE;

CREATE CONSTRAINT maintenance_action_id IF NOT EXISTS
FOR (a:MaintenanceAction) REQUIRE a.action_id IS UNIQUE;

CREATE CONSTRAINT failure_mode_name IF NOT EXISTS
FOR (fm:FailureMode) REQUIRE fm.name IS UNIQUE;

CREATE CONSTRAINT technician_id IF NOT EXISTS
FOR (t:Technician) REQUIRE t.technician_id IS UNIQUE;

CREATE CONSTRAINT spare_part_id IF NOT EXISTS
FOR (p:SparePart) REQUIRE p.part_id IS UNIQUE;

// Useful lookup indexes (non-unique)
CREATE INDEX fault_event_timestamp IF NOT EXISTS
FOR (f:FaultEvent) ON (f.timestamp);

CREATE INDEX fault_event_status IF NOT EXISTS
FOR (f:FaultEvent) ON (f.status);

// ============================================================
// NODE LABELS
// ============================================================
// (:Machine {machine_id, name, location, install_date})
// (:Sensor  {sensor_id, name, group, unit})
//     group in {pressure, motor, flow, temperature, other}
// (:FaultEvent {
//     event_id, timestamp, confidence, source ("pytorch"|"onnx"),
//     status ("open"|"resolved"|"false_positive"),
//     drift_score, model_version
// })
// (:MaintenanceAction {action_id, description, performed_at, outcome})
// (:FailureMode {name, description})
// (:Technician {technician_id, name})
// (:SparePart {part_id, name})

// ============================================================
// RELATIONSHIP TYPES
// ============================================================
// (:Sensor)-[:MONITORS]->(:Machine)
// (:FaultEvent)-[:OCCURRED_ON]->(:Machine)
// (:FaultEvent)-[:FLAGGED_BY {importance: float, rank: int}]->(:Sensor)
//     -- one edge per top-N sensor from the attention explainability output
// (:FaultEvent)-[:MATCHES {similarity: float}]->(:FailureMode)
// (:FaultEvent)-[:RESOLVED_BY]->(:MaintenanceAction)
// (:MaintenanceAction)-[:PERFORMED_BY]->(:Technician)
// (:MaintenanceAction)-[:USED]->(:SparePart)
// (:FaultEvent)-[:SIMILAR_TO {score: float}]->(:FaultEvent)
//     -- computed offline / on demand from sensor-importance vector similarity

// ============================================================
// SEED DATA -- one machine + its 17 sensors
// (matches the 17 hydraulic sensors used by the TFT model)
// ============================================================
MERGE (m:Machine {machine_id: "HYD-01"})
ON CREATE SET m.name = "Hydraulic Test Rig 1", m.location = "Line 1", m.install_date = date("2020-01-01");

MATCH (m:Machine {machine_id: "HYD-01"})
UNWIND [
  {id:"PS1", group:"pressure", unit:"bar"}, {id:"PS2", group:"pressure", unit:"bar"},
  {id:"PS3", group:"pressure", unit:"bar"}, {id:"PS4", group:"pressure", unit:"bar"},
  {id:"PS5", group:"pressure", unit:"bar"}, {id:"PS6", group:"pressure", unit:"bar"},
  {id:"EPS1", group:"motor", unit:"W"},
  {id:"FS1", group:"flow", unit:"L/min"}, {id:"FS2", group:"flow", unit:"L/min"},
  {id:"TS1", group:"temperature", unit:"C"}, {id:"TS2", group:"temperature", unit:"C"},
  {id:"TS3", group:"temperature", unit:"C"}, {id:"TS4", group:"temperature", unit:"C"},
  {id:"VS1", group:"other", unit:"mm/s"}, {id:"CE", group:"other", unit:"%"},
  {id:"CP", group:"other", unit:"bar"}, {id:"SE", group:"other", unit:"%"}
] AS sensor
MERGE (s:Sensor {sensor_id: sensor.id})
ON CREATE SET s.group = sensor.group, s.unit = sensor.unit
MERGE (s)-[:MONITORS]->(m);
