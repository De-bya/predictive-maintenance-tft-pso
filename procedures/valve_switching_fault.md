# Procedure: Valve Switching Behavior Anomaly (maps to dataset 'valve' label)

**Failure mode:** Valve condition percentages 73/80/90/100 — lower values
indicate degraded switching behavior.

**Typical signature:** Irregular, high-frequency oscillation in pressure
sensors immediately downstream of the valve, or slower-than-expected pressure
rise/fall time during a switching event.

**First checks:**
1. Check valve solenoid response time against baseline spec.
2. Inspect for contamination in the valve spool (see contamination procedure).
3. Verify control signal timing from the PLC/controller is within spec.
