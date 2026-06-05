import time

class SREMonitor:
    """Tracks SLOs and SLIs for the predictive maintenance system"""

    def __init__(self, availability_target=0.973):
        self.availability_target = availability_target
        self.uptime   = 0.0
        self.downtime = 0.0
        self._start   = None

    def start_window(self):
        self._start = time.time()

    def record_uptime(self):
        if self._start:
            self.uptime += time.time() - self._start

    def record_downtime(self):
        if self._start:
            self.downtime += time.time() - self._start

    def current_availability(self):
        total = self.uptime + self.downtime
        if total == 0:
            return 1.0
        return self.uptime / total

    def slo_breached(self):
        avail    = self.current_availability()
        breached = avail < self.availability_target
        status   = "❌ BREACHED" if breached else "✅ OK"
        print(f"SLO status: {status} | "
              f"Availability: {avail*100:.2f}% "
              f"(target: {self.availability_target*100:.1f}%)")
        return breached