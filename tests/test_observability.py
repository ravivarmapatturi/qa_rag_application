import os

import observability


def test_trace_config_no_keys_returns_plain_config(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    observability._handler = None
    observability._warned = False

    config = observability.trace_config()

    assert "callbacks" not in config


def test_trace_config_with_keys_attaches_callback(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    observability._handler = None

    config = observability.trace_config()

    assert "callbacks" in config
    assert len(config["callbacks"]) == 1


def test_trace_config_preserves_extra_kwargs(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    observability._handler = None

    config = observability.trace_config(run_name="my run")

    assert config["run_name"] == "my run"
