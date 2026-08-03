# Procedure: What It Means When the System Reports Drift

**Context:** The system uses a CUSUM (cumulative sum) statistical test to
detect when live predictions start deviating from expected model behavior.
This does not mean the machine is failing more — it means the *model's*
behavior on new data looks statistically different from what it was
validated on.

**What happens automatically:** If enough drift points accumulate, the
system automatically retrains a new model and only promotes it to
production if it measurably outperforms the current one on the held-out
test set. Rejected retrains are logged but do not change production
behavior.

**Operator action:** Drift alerts are informational for maintenance staff —
no physical inspection is required based on a drift alert alone. If drift
is accompanied by a spike in FaultEvent frequency for a specific machine,
treat that as a signal to schedule a physical inspection of that machine
specifically, since it may indicate a real, gradual mechanical change (e.g.
wear) rather than a data quality issue.
