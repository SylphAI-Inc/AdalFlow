"""Simulated noisy evolution: what does ``evolve(confirm=...)`` buy at a fixed budget?

No model is called. A candidate is its true quality; a proposal perturbs the parent's
quality (most mutations are neutral or slightly harmful), and the evaluator returns
the true quality plus Gaussian noise. For each noise level and ``confirm`` setting the
script reports, over many runs, the true quality of the candidate ``evolve`` returns
and how far its reported score overstates it (the winner's curse).

    python benchmarks/evolution/noise_sim.py --runs 200 --budget 60
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adalflow"))

from adalflow.optim.evolution import BEST, Proposal, evolve  # noqa: E402


def quality(text: str) -> float:
    return float(text.split("=", 1)[1].split()[0])


class DriftProposer:
    """child quality = parent quality + N(mean_step, step_sd)."""

    def __init__(self, rng: random.Random, mean_step: float, step_sd: float) -> None:
        self.rng, self.mean_step, self.step_sd = rng, mean_step, step_sd
        self.count = 0

    def propose(self, parent, inspiration=None) -> Proposal:
        self.count += 1
        q = quality(parent.text) + self.rng.gauss(self.mean_step, self.step_sd)
        return Proposal(text=f"q={q:.6f} #{self.count}")


def one(run: int, sigma: float, confirm: int, args) -> tuple:
    noise = random.Random(10_000 + run)

    def evaluate(text: str) -> float:
        return quality(text) + noise.gauss(0.0, sigma)

    res = evolve(
        "q=0.5",
        evaluate,
        DriftProposer(random.Random(run), args.mean_step, args.step_sd),
        max_evaluations=args.budget,
        selection=BEST,
        random_seed=run,
        max_stalls=args.budget,
        confirm=confirm,
    )
    true_q = quality(res.best.text)
    return true_q, res.best.score - true_q, len(res.archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--mean-step", type=float, default=-0.01)
    parser.add_argument("--step-sd", type=float, default=0.03)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.02, 0.05, 0.1])
    parser.add_argument("--confirms", type=int, nargs="+", default=[0, 1, 2, 4])
    args = parser.parse_args()

    print(
        f"budget {args.budget}, {args.runs} runs per cell, mutation "
        f"N({args.mean_step}, {args.step_sd}), seed quality 0.5\n"
    )
    print("| noise sd | confirm | true quality of best | overstatement | candidates |")
    print("|---|---|---|---|---|")
    for sigma in args.sigmas:
        for confirm in args.confirms:
            rows = [one(r, sigma, confirm, args) for r in range(args.runs)]
            q, gap, n = (list(x) for x in zip(*rows))
            se = statistics.stdev(q) / len(q) ** 0.5
            print(
                f"| {sigma} | {confirm} | {statistics.mean(q):.4f} ± {se:.4f} "
                f"| {statistics.mean(gap):+.4f} | {statistics.mean(n):.1f} |"
            )


if __name__ == "__main__":
    main()
