"""Shared LOCAL_TEST_THROTTLE contract every loop script codes against.

LOCAL_TEST_THROTTLE and THROTTLE_LIMIT come from llm_client.config, not from
each script; every loop script applies apply_throttle() rather than
declaring its own cap.
"""
import llm_client.config as config

LOCAL_TEST_THROTTLE: bool = config.LOCAL_TEST_THROTTLE


def throttle_limit_clause() -> str:
    return f"LIMIT {config.THROTTLE_LIMIT}"


def apply_throttle(items: list) -> list:
    if LOCAL_TEST_THROTTLE:
        return items[: config.THROTTLE_LIMIT]
    return items
