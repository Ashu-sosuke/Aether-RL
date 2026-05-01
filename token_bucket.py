import time

class TokenBucket:
    def __init__(self, capacity: int = 100, refill_rate: float = 10.0):
        self.capacity     = capacity
        self.refill_rate  = refill_rate   # tokens per minute
        self.current      = float(capacity)
        self._last_refill = time.monotonic()

    def _refill(self):
        now     = time.monotonic()
        elapsed = now - self._last_refill          # seconds
        added   = elapsed * (self.refill_rate / 60.0)
        self.current = min(self.capacity, self.current + added)
        self._last_refill = now

    def consume(self, amount: int = 1) -> bool:
        self._refill()
        if self.current >= amount:
            self.current -= amount
            return True
        return False

    def balance(self) -> int:
        self._refill()
        return int(self.current)

    def reset_at_seconds(self) -> float:
        needed = 1.0
        deficit = needed - self.current
        if deficit <= 0: return 0.0
        return deficit / (self.refill_rate / 60.0)

# Cost table
ACTION_COST = 1    # each ActionCommand sent to Android
LLM_COST    = 3    # each LLM API call
