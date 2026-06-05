import time

def with_exponential_backoff(fn, initial_delay=1.0, max_retries=5):
    """
    Equation 7 from the paper:
    backoff_t = initial_delay x 2^retry_count
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            backoff = initial_delay * (2 ** attempt)
            print(f"⚠️  Attempt {attempt+1} failed: {e}")
            print(f"    Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
    raise RuntimeError("❌ Max retries exceeded — operation failed")