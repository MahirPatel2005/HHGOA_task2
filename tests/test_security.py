from app.security import SlidingWindowRateLimiter, public_error, sanitize_error


def test_sliding_window_rate_limiter_blocks_after_limit():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("client-a")
    assert limiter.allow("client-a")
    assert not limiter.allow("client-a")
    assert limiter.allow("client-b")


def test_error_sanitizer_removes_provider_credentials():
    message = "Bearer gsk_secret_value api-subscription-key=sk_secret_value"
    safe = sanitize_error(message)
    assert "gsk_secret_value" not in safe
    assert "sk_secret_value" not in safe
    assert "<redacted>" in safe


def test_public_error_is_stable_and_secret_free():
    assert public_error("provider_failed") == "provider_failed"
