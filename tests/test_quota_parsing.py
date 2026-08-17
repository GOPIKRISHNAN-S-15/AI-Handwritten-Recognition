"""Unit tests for GeminiService quota / retry-delay parsing in _record_probe_error."""
import json
import sys

sys.path.insert(0, ".")
from genai.ai_service import GeminiService


def make_429_payload(limit=20, retry_delay="927.592s"):
    return {
        "error": {
            "code": 429,
            "message": "RESOURCE_EXHAUSTED (Quota exceeded for quota metric 'generativelanguage.googleapis.com/fd_req_free_tier_requests' and limit '20' of service 'generativelanguage.googleapis.com' for consumer 'project_number:123'.)",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "RATE_LIMIT_EXCEEDED",
                    "domain": "googleapis.com",
                    "metadata": {"service": "generativelanguage.googleapis.com"},
                },
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaMetric": "generativelanguage.googleapis.com/fd_req_free_tier_requests",
                            "quotaValue": limit,
                            "subject": "project:123",
                        }
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
            ],
        }
    }


def new_svc():
    svc = GeminiService.__new__(GeminiService)
    svc.status = "uninitialized"
    svc.is_available = False
    svc.quota_limit = None
    svc.retry_after_seconds = None
    return svc


def test_decimal_duration():
    svc = new_svc()
    svc._record_probe_error(json.dumps(make_429_payload()))
    assert svc.status == "quota_limited", svc.status
    assert svc.quota_limit == 20, svc.quota_limit
    assert svc.retry_after_seconds == 927, svc.retry_after_seconds
    print("PASS test_decimal_duration")


def test_integer_duration():
    svc = new_svc()
    svc._record_probe_error(json.dumps(make_429_payload(retry_delay="60s")))
    assert svc.retry_after_seconds == 60, svc.retry_after_seconds
    print("PASS test_integer_duration")


def test_different_limit():
    svc = new_svc()
    svc._record_probe_error(json.dumps(make_429_payload(limit=50)))
    assert svc.quota_limit == 50, svc.quota_limit
    print("PASS test_different_limit")


def test_non_json_fallback():
    svc = new_svc()
    svc._record_probe_error("some 500 network error 429 message")
    assert svc.status == "quota_limited"
    assert svc.quota_limit is None
    assert svc.retry_after_seconds is None
    print("PASS test_non_json_fallback")


def test_404_classification():
    svc = new_svc()
    svc._record_probe_error('404 Not Found. The model gemini-1.5-flash was not found.')
    assert svc.status == "model_unavailable", svc.status
    print("PASS test_404_classification")


def test_401_classification():
    svc = new_svc()
    svc._record_probe_error("401 API key not valid")
    assert svc.status == "key_missing", svc.status
    print("PASS test_401_classification")


class FakeResponse:
    """Minimal stand-in for an HTTP response with JSON body."""

    def __init__(self, payload, status_code=429):
        self.text = json.dumps(payload)
        self.status_code = status_code


class FakeClientError(Exception):
    """Minimal stand-in for google.genai.errors.ClientError."""

    def __init__(self, response, status=429):
        super().__init__(f"{status} Resource exhausted")
        self.response = response
        self.status = status


def test_client_error_object_form():
    svc = new_svc()
    err = FakeClientError(FakeResponse(make_429_payload(limit=20)))
    svc._record_probe_error(err)
    assert svc.status == "quota_limited", svc.status
    assert svc.quota_limit == 20, svc.quota_limit
    assert svc.retry_after_seconds == 927, svc.retry_after_seconds
    print("PASS test_client_error_object_form")


def test_client_error_404_object_form():
    svc = new_svc()
    payload = {"error": {"code": 404, "message": "Model not found", "status": "NOT_FOUND"}}
    err = FakeClientError(FakeResponse(payload, 404), status=404)
    svc._record_probe_error(err)
    assert svc.status == "model_unavailable", svc.status
    print("PASS test_client_error_404_object_form")


if __name__ == "__main__":
    test_decimal_duration()
    test_integer_duration()
    test_different_limit()
    test_non_json_fallback()
    test_404_classification()
    test_401_classification()
    test_client_error_object_form()
    test_client_error_404_object_form()
    print("All quota parsing tests passed.")
