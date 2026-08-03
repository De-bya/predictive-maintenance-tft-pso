# Procedure: Full Sensor Dropout / Missing Data

**Failure mode:** A sensor channel goes fully offline (zero or NaN readings
across the window) rather than showing an anomalous but present value.

**Typical signature:** The system's chaos-engineering tests (see project
resilience report) showed that dropping a single pressure sensor (PS1)
degrades model accuracy from 96.5% to about 90.9% — the model keeps
predicting but with meaningfully reduced confidence, rather than failing
outright.

**First checks:**
1. Treat this as a wiring/connector issue first, not a hydraulic fault —
   check the physical connection and power to the sensor.
2. If the sensor cannot be restored quickly, flag any prediction made during
   the dropout window as lower-confidence in the operator log.
3. Once restored, verify the sensor's first few post-restoration readings
   are stable before trusting new predictions at full confidence.
