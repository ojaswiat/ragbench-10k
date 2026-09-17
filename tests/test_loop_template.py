import importlib

import llm_client.config as config
import loop_template


def test_throttle_limit_clause_matches_config():
    assert loop_template.throttle_limit_clause() == f"LIMIT {config.THROTTLE_LIMIT}"


def test_apply_throttle_caps_at_three_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_TEST_THROTTLE", True)
    importlib.reload(loop_template)
    items = list(range(10))
    assert loop_template.apply_throttle(items) == [0, 1, 2]


def test_apply_throttle_passthrough_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_TEST_THROTTLE", False)
    importlib.reload(loop_template)
    items = list(range(10))
    assert loop_template.apply_throttle(items) == items
