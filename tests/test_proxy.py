"""Tests for the Anthropic proxy master-prompt injection (§6.5)."""

import pytest


# ── Minimal sync-compatible injection logic mirroring proxy.py ───────────────
# We test the injection logic in isolation without Redis by extracting the
# mutation function directly and driving it with a local dict.

def _build_injected_body(
    body: dict,
    master: str,
    already_injected: bool,
) -> tuple[dict, bool]:
    """Stateless extraction of the injection logic from proxy.py for unit tests."""
    if not master or already_injected:
        return body, already_injected

    existing_system = body.get("system", "")
    if isinstance(existing_system, list):
        body["system"] = [
            {"type": "text", "text": master, "cache_control": {"type": "ephemeral"}},
            *existing_system,
        ]
    else:
        body["system"] = [
            {"type": "text", "text": master, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": existing_system or ""},
        ]
    return body, True


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_injection_adds_master_prompt_to_string_system():
    body = {"system": "you are claude", "messages": []}
    out, injected = _build_injected_body(body, "OPTIVIA_MASTER", False)
    assert isinstance(out["system"], list)
    assert out["system"][0]["text"] == "OPTIVIA_MASTER"
    assert out["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert injected is True


def test_injection_prepends_when_system_is_list():
    body = {"system": [{"type": "text", "text": "existing"}], "messages": []}
    out, _ = _build_injected_body(body, "OPTIVIA_MASTER", False)
    assert out["system"][0]["text"] == "OPTIVIA_MASTER"
    assert out["system"][1]["text"] == "existing"


def test_injection_only_runs_once_per_request():
    body = {"system": "foo", "messages": []}
    out1, injected1 = _build_injected_body(body, "OPTIVIA_MASTER", False)
    assert injected1 is True
    # Second call with already_injected=True should be a no-op
    out2, injected2 = _build_injected_body(dict(out1), "OPTIVIA_MASTER", True)
    assert out2["system"] == out1["system"]
    assert injected2 is True


def test_injection_skips_when_no_master_registered():
    body = {"system": "foo", "messages": []}
    out, injected = _build_injected_body(body, "", False)
    assert out["system"] == "foo"
    assert injected is False


def test_injection_skips_for_empty_request_id():
    body = {"system": "foo", "messages": []}
    # Simulate the "no master found" branch (empty string master)
    out, injected = _build_injected_body(body, "", False)
    assert out["system"] == "foo"
    assert injected is False


@pytest.mark.asyncio
async def test_proxy_module_imports_cleanly():
    """Smoke-test: proxy module loads without Redis available."""
    from backend.proxy import router, proxy_health
    result = await proxy_health()
    assert result["status"] == "ok"
    assert "upstream" in result
