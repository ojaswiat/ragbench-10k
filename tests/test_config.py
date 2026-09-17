import llm_client.config as config

def test_model_routing_has_all_stages():
    expected_stages = {"generator", "critic", "p3_index_build", "answerer", "judge", "debug"}
    assert set(config.MODEL_ROUTING.keys()) == expected_stages

def test_generator_and_critic_are_different_families():
    gen = config.MODEL_ROUTING["generator"]
    crit = config.MODEL_ROUTING["critic"]
    # Each entry is dict with model and provider
    gen_family = gen["model"].split("/")[0] if isinstance(gen, dict) else gen.split("/")[0]
    crit_family = crit["model"].split("/")[0] if isinstance(crit, dict) else crit.split("/")[0]
    assert gen_family != crit_family

def test_answerer_and_judge_are_different_families():
    answerer = config.MODEL_ROUTING["answerer"]
    judge = config.MODEL_ROUTING["judge"]
    answerer_family = answerer["model"].split("-")[0] if isinstance(answerer, dict) else answerer.split("-")[0]
    judge_family = judge["model"].split("/")[0] if isinstance(judge, dict) else judge.split("/")[0]
    assert answerer_family != judge_family

def test_throttle_limit_is_three():
    assert config.THROTTLE_LIMIT == 3

def test_concurrency_cap_is_five():
    assert config.GROQ_MAX_CONCURRENCY == 5

def test_answerer_and_judge_route_to_openrouter_with_pinned_provider():
    """Both benchmark stages run on OpenRouter, pinned to one upstream host.

    OpenRouter load-balances across hosts that differ in quantisation, so an
    unpinned run would mix fp8 and bf16 outputs across the 900 cells and
    break the single-temp-0-run-per-cell reproducibility claim.
    """
    for stage in ("answerer", "judge"):
        entry = config.MODEL_ROUTING[stage]
        assert entry["provider"] == "openrouter", stage
        pin = entry["extra_body"]["provider"]
        assert pin["allow_fallbacks"] is False, stage
        assert len(pin["order"]) == 1, stage


def test_answerer_and_judge_remain_different_families():
    """Guardrails §2: no model may grade its own output."""
    answerer = config.MODEL_ROUTING["answerer"]["model"]
    judge = config.MODEL_ROUTING["judge"]["model"]
    assert "llama" in answerer.lower()
    assert "qwen" in judge.lower()


def test_every_openrouter_stage_caps_reasoning():
    """A reasoning-tuned model with no cap can emit an unbounded chain and hang
    the batch behind it. The judge lost its cap when it moved from Groq to
    OpenRouter and stalled a 60-row run for 15 minutes with no error, so this
    asserts the invariant rather than naming the stages that happened to have
    caps at the time."""
    for stage, entry in config.MODEL_ROUTING.items():
        if entry["provider"] != "openrouter":
            continue
        reasoning = (entry.get("extra_body") or {}).get("reasoning")
        assert reasoning is not None, f"{stage} routes to OpenRouter with no reasoning cap"
        assert reasoning.get("effort") in {"low", "medium", "high"}, stage
