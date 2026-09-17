# Groq Rate Limits — Verified

Checked: 2026-07-21 (update this date whenever re-verified; limits are account-tier-dependent and change without notice).

All 5 model routes verified live and reachable on this account (`generator`, `critic`, `p3_index_build`, `answerer`, `judge`). Note: `qwen/qwen3-32b` (originally specced for Critic/Judge) returned `404 model_not_found` — Groq no longer hosts it on this account. Swapped to `qwen/qwen3.6-27b`; re-verified OK after the swap.

Captured from live `x-ratelimit-*` response headers (per-request snapshot, not the console dashboard):

| Stage | Model | requests limit (window) | tokens limit (window) |
|---|---|---|---|
| Generator | openai/gpt-oss-120b | 1000 | 8000 |
| Critic | qwen/qwen3.6-27b | 1000 | 8000 |
| P3 index build | llama-3.1-8b-instant | 14400 | 6000 |
| Answerer | llama-3.3-70b-versatile | 1000 | 12000 |
| Judge | qwen/qwen3.6-27b | 1000 | 8000 |

**Not yet verified — needs a human console visit:** the response headers don't label which window (RPM vs RPD, TPM vs TPD) each number belongs to, and don't expose TPD at all. Full RPM/RPD/TPM/TPD split requires visiting https://console.groq.com/settings/limits directly (not exposed via API). This is a [PASSIVE] gap — later phases can start without it, but Phase 7's 900-run pacing needs the real TPD ceiling before the full matrix launches.

Notes:
- Limits apply **per organization**, not per key — extra keys do not raise the ceiling (`Guardrails.md` §5).
- The Answerer (`Llama 3.3 70B`) has the largest TPM observed (12000) but is the TPD-pacing bottleneck for Phase 7's 900-run matrix (per `Budget.md`); pacing should be built around whichever model proves tightest on TPD once confirmed.
- Re-check at https://console.groq.com/settings/limits before Phase 7 kicks off — free-tier limits are revised periodically.
