# Procedure: Pressure Sensor (PS1–PS6) Anomaly

**Failure mode:** Abnormal pressure readings or sudden dropout on any PS1–PS6 channel.

**Typical signature:** High attention weight on a single PS sensor group, sustained
deviation from the sensor's rolling mean, or a flatline (zero-variance) reading.

**First checks (in order):**
1. Verify physical sensor connector and cable for corrosion or looseness.
2. Check for air trapped in the hydraulic line near the sensor tap point.
3. Compare the flagged sensor's raw reading against its redundant neighbor
   (e.g. PS1 vs PS2) — a single-sensor divergence points to a sensor fault
   rather than a real hydraulic fault.
4. If the reading is a flatline at exactly 0, suspect a disconnected sensor
   rather than an actual pressure loss.

**Escalation:** If the sensor checks out physically, escalate to hydraulic
technician to inspect the pump and accumulator for real pressure loss.
