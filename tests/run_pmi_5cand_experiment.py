from __future__ import annotations

import math
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tandem import TandemEngine, load_amboss_questions
from tandem.engine.candidate_scorer import CandidateDifferentialScorer

AMBOSS_QUESTIONS_DIR = Path(
    "/Users/rohanmaster/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Resources/data/raw/amboss_qbank/questions"
)


def run_experiment():
    print("=" * 80)
    print("5-CANDIDATE EMPIRICAL EXPERIMENT: RAW MEAN-LOGP vs CONTRASTIVE PMI")
    print("=" * 80)

    items = load_amboss_questions(AMBOSS_QUESTIONS_DIR, max_questions=1)
    assert len(items) >= 1
    item = items[0]

    vignette = item.prompt
    candidates = item.metadata.get("all_candidates")
    gold = item.ground_truth

    print(f"Amboss QID:      {item.metadata.get('qid')}")
    print(f"Gold Target:     '{gold}'")
    print(f"Candidate Count: {len(candidates)}")
    for i, c in enumerate(candidates, 1):
        is_gold = " (GOLD)" if c == gold else ""
        print(f"   Option {i}: '{c}'{is_gold}")

    engine = TandemEngine()
    scorer = CandidateDifferentialScorer(engine=engine)

    # 1. Baseline: Raw Mean Logprob (no neutral context)
    print("\n" + "-" * 80)
    print("1. BASELINE: RAW MEAN-LOGP SCORING")
    print("-" * 80)
    res_raw = scorer.score_candidates(
        vignette=vignette,
        candidates=candidates,
        prefix_template="\n<answer>",
        neutral_context=None,
    )
    print(f"Differential Entropy: {res_raw.entropy_bits:.3f} bits")
    print(f"{'Rank':>4} | {'Candidate':<38} | {'Mean Logp':>10} | {'Prob Mass':>10} | {'Tokens':>6}")
    print("-" * 80)
    for c in res_raw.candidates:
        mark = " *" if c.candidate == gold else "  "
        print(f"{c.rank:4d} | {c.candidate:<38} | {c.mean_logprob:10.4f} | {c.candidate_probability*100:9.2f}% | {len(c.token_ids):6d}{mark}")

    # 2. Contrastive PMI: with generic clinical prompt
    neutral_context = "A patient presents with symptoms.\n<answer>"
    print("\n" + "-" * 80)
    print(f"2. CONTRASTIVE PMI SCORING (Neutral context: {repr(neutral_context)})")
    print("-" * 80)
    res_pmi = scorer.score_candidates(
        vignette=vignette,
        candidates=candidates,
        prefix_template="\n<answer>",
        neutral_context=neutral_context,
    )
    print(f"Differential Entropy: {res_pmi.entropy_bits:.3f} bits")
    print(f"{'Rank':>4} | {'Candidate':<38} | {'PMI Score':>10} | {'Mean Logp':>10} | {'Prob Mass':>10} | {'Tokens':>6}")
    print("-" * 80)
    for c in res_pmi.candidates:
        mark = " *" if c.candidate == gold else "  "
        print(f"{c.rank:4d} | {c.candidate:<38} | {c.pmi_score:10.4f} | {c.mean_logprob:10.4f} | {c.candidate_probability*100:9.2f}% | {len(c.token_ids):6d}{mark}")

    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
