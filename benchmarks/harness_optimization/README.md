# Harness optimization benchmark: held-out tasks

`run.py` checks whether a harness refined without labels (`HarnessOptimizer.run_domain`)
does better on held-out tasks than the starting harness and than a one-shot prompt
rewrite. Labels are used only to score the held-out runs (`HarnessOptimizer.evaluate`).

The domain: an agent answers account questions with a paginated `list_transactions`
tool (5 items per page, `next_cursor`) and a `calculator`. The base harness is stale:
it says the tool returns every transaction in one call. Training tasks ask for a
total, a count over a threshold and a refund total on 3 accounts; held-out tasks use
4 other accounts and add a question type training never asks (largest transaction).

| condition | what it is |
|---|---|
| `base` | the stale harness |
| `prompt_only` | the editor model rewrites the prompt once from the tool docs and the training task texts, without running the agent |
| `refined` | the harness `run_domain` selected on the last training task (k=1, parallel) |
| `hand_fixed` | the stale sentence replaced with a correct one (reference) |

```bash
python benchmarks/harness_optimization/run.py --provider anthropic \
    --agent-model claude-haiku-4-5-20251001 --editor-model claude-haiku-5-5 \
    --trials 3 --repeats 4 --out runs/harness_bench
```

## Results (2026-10-21)

Agent `claude-haiku-4-5-20251001`, editor / selector / reflector `claude-haiku-5-5`,
`max_steps=8`. Held-out accuracy over 4 tasks x 4 rollouts; mean [min, max] over 3
independent trials (`base` and `hand_fixed` do not change between trials, so they
were measured once).

| condition | runs | held-out accuracy |
|---|---|---|
| base | 1 | 0.938 |
| prompt_only | 3 | 0.562 [0.312, 1.000] |
| refined | 3 | 1.000 [1.000, 1.000] |
| hand_fixed | 1 | 1.000 |

What this does and does not show:

- `refined` vs `base` is not a measurable gain here. With 8 steps the agent usually
  notices `next_cursor` on its own and the stale harness costs little (15/16
  correct). In one trial `run_domain` kept the base harness unchanged and that
  harness scored 16/16, so the gap is within rollout noise.
- Rewriting the prompt without trajectories hurt: in 2 of 3 trials the rewrite made
  the stale claim stronger ("call list_transactions once ... do not request pages")
  and held-out accuracy fell to 0.31 and 0.38. Refinement from trajectories never
  made the harness worse, and in 2 of 3 trials it replaced the stale claim with a
  correct description of the pagination (the evidence is the non-null
  `next_cursor` in the probe trajectories).
- In a quick check with `max_steps=5` the stale harness was right 4/9 times, but
  that setting also fails correct harnesses on 20-item accounts (4 pages, a
  calculator call and the answer), so it was not used for the comparison.
- Small domain, one model pair, 3 trials. In quick checks (9 rollouts each) a
  stronger agent, claude-haiku-5-5, followed `next_cursor` despite the stale claim
  (9/9), and handled easier variants (an optional `page_size`, 30-45 item sums) with
  a neutral prompt (9/9); claude-haiku-4-5 got 8/9 on 60-90 item sums. Harness flaws
  a capable agent can see for itself leave little for refinement to fix.
