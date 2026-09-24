from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tandem import (
    CandidateDifferentialScore,
    CandidateDifferentialScorer,
    DifferentialScoringResult,
    UMLSEntityTrie,
    UMLSTrieNode,
)


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<answer>": 3,
            "acute": 10,
            "cholecystitis": 11,
            "appendicitis": 12,
            "pneumonia": 13,
            "streptococcus": 14,
            "aortic": 15,
            "dissection": 16,
            "pain": 17,
            "fever": 18,
            "cough": 19,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(1)
        # Simple whitespace/punctuation splitting
        clean = text.replace("<answer>", " <answer> ").replace("\n", " ").lower()
        for w in clean.split():
            tokens.append(self.vocab.get(w, 99))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return " ".join([self.inv_vocab.get(t, "<unk>") for t in token_ids])


class MockDynamicCache:
    def __init__(self, initial_len: int = 0):
        self._seq_len = initial_len

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len

    def crop(self, max_length: int) -> int:
        self._seq_len = max_length
        return self._seq_len


class MockRunner:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.device = torch.device("cpu")
        self.crop_calls: List[int] = []

    def forward_causal(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[MockDynamicCache] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, MockDynamicCache]:
        B, seq_len = input_ids.shape
        vocab_size = 120

        # Create predictable synthetic logits
        logits = torch.zeros(B, seq_len, vocab_size, dtype=torch.float32)

        # Give higher logits to tokens: cholecystitis (11) and acute (10)
        logits[:, :, 10] = 5.0  # acute
        logits[:, :, 11] = 6.0  # cholecystitis
        logits[:, :, 12] = 2.0  # appendicitis
        logits[:, :, 13] = 1.0  # pneumonia

        if past_key_values is None:
            cache = MockDynamicCache(initial_len=seq_len)
        else:
            cache = past_key_values
            cache._seq_len += seq_len

        return logits, cache


class MockEngine:
    def __init__(self):
        self.runner = MockRunner()


def test_candidate_scorer_synthetic_ranking_and_normalization():
    """Verify that CandidateDifferentialScorer ranks candidates by length-normalized mean logp."""
    engine = MockEngine()
    scorer = CandidateDifferentialScorer(engine=engine)

    vignette = "Patient presents with acute right upper quadrant pain.\n<answer>"
    candidates = [
        "acute cholecystitis",  # Expected high likelihood
        "acute appendicitis",    # Expected moderate likelihood
        "pneumonia",             # Expected lower likelihood
    ]

    res = scorer.score_candidates(vignette, candidates)

    assert isinstance(res, DifferentialScoringResult)
    assert len(res.candidates) == 3
    assert res.top_candidate.candidate == "acute cholecystitis"
    assert res.top_candidate.rank == 1

    # Verify length normalization
    assert res.candidates[0].mean_logprob > res.candidates[1].mean_logprob
    assert res.candidates[1].mean_logprob > res.candidates[2].mean_logprob

    # Verify probability distribution sums to 1.0
    total_prob = sum(c.candidate_probability for c in res.candidates)
    assert total_prob == pytest.approx(1.0, abs=1e-5)

    # Verify entropy is positive and bounded by log2(3) = 1.585
    assert 0.0 <= res.entropy_bits <= math.log2(3)


def test_candidate_scorer_contrastive_pmi():
    """Verify that contrastive PMI scoring computes delta against neutral context and ranks candidates."""
    engine = MockEngine()
    scorer = CandidateDifferentialScorer(engine=engine)

    vignette = "Patient presentation with fever.\n<answer>"
    candidates = ["acute cholecystitis", "pneumonia"]
    neutral = "The most likely diagnosis is:"

    res = scorer.score_candidates(vignette, candidates, neutral_context=neutral)

    assert isinstance(res, DifferentialScoringResult)
    for c in res.candidates:
        assert c.pmi_score is not None
        assert isinstance(c.pmi_score, float)

    # In MockRunner logits are identical across contexts, so PMI delta is near 0.0
    for c in res.candidates:
        assert abs(c.pmi_score) < 1e-4

    # Ranking is preserved and probability sums to 1.0
    total_prob = sum(c.candidate_probability for c in res.candidates)
    assert total_prob == pytest.approx(1.0, abs=1e-5)


def test_candidate_scorer_umls_trie_resolution():
    """Verify that candidates registered in UMLSEntityTrie resolve CUIs and concepts."""
    engine = MockEngine()
    trie = UMLSEntityTrie()

    # Insert known concepts
    # acute (10) + cholecystitis (11)
    trie.insert([10, 11], concept_name="Acute cholecystitis", cui="C0008325")
    # acute (10) + appendicitis (12)
    trie.insert([10, 12], concept_name="Acute appendicitis", cui="C0003615")

    scorer = CandidateDifferentialScorer(engine=engine, umls_trie=trie)

    vignette = "Right upper quadrant abdominal tenderness.\n<answer>"
    candidates = ["acute cholecystitis", "acute appendicitis", "pneumonia"]

    res = scorer.score_candidates(vignette, candidates)

    # Verify UMLS resolution
    c0 = res.candidates[0]
    assert c0.candidate == "acute cholecystitis"
    assert c0.umls_verified is True
    assert c0.cui == "C0008325"

    c1 = res.candidates[1]
    assert c1.candidate == "acute appendicitis"
    assert c1.umls_verified is True
    assert c1.cui == "C0003615"

    # Pneumonia was not inserted in trie
    c2 = res.candidates[2]
    assert c2.candidate == "pneumonia"
    assert c2.umls_verified is False
    assert c2.cui is None


def test_candidate_scorer_entropy_extremes():
    """Verify entropy behaves correctly under certainty vs uniform distributions."""
    engine = MockEngine()
    scorer = CandidateDifferentialScorer(engine=engine)

    # When temperature is very small (near 0), distribution peaks -> entropy -> 0
    res_cold = scorer.score_candidates(
        "Vignette\n<answer>",
        ["acute cholecystitis", "pneumonia"],
        temperature=0.01,
    )
    assert res_cold.entropy_bits < 0.05
    assert res_cold.candidates[0].candidate_probability > 0.99

    # When temperature is high, distribution flattens -> entropy -> log2(2) = 1.0
    res_hot = scorer.score_candidates(
        "Vignette\n<answer>",
        ["acute cholecystitis", "pneumonia"],
        temperature=100.0,
    )
    assert res_hot.entropy_bits == pytest.approx(1.0, abs=0.05)


def test_candidate_scorer_summary_output():
    """Verify that DifferentialScoringResult.summary() outputs structured clinical text."""
    engine = MockEngine()
    scorer = CandidateDifferentialScorer(engine=engine)

    res = scorer.score_candidates("RUQ Pain\n<answer>", ["acute cholecystitis", "pneumonia"])
    summary_text = res.summary()

    assert "Vignette:" in summary_text
    assert "Top Candidate:" in summary_text
    assert "Differential Entropy:" in summary_text
    assert "#1: 'acute cholecystitis'" in summary_text


AMBOSS_QUESTIONS_DIR = Path(
    "/Users/rohanmaster/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Resources/data/raw/amboss_qbank/questions"
)


@pytest.mark.integration
def test_candidate_scorer_real_weights_amboss():
    """Verify CandidateDifferentialScorer evaluates real Amboss options on Apple Silicon MPS."""
    if not AMBOSS_QUESTIONS_DIR.exists():
        pytest.skip("Amboss questions directory not found on disk")

    from tandem import TandemEngine, load_amboss_questions

    questions = load_amboss_questions(AMBOSS_QUESTIONS_DIR, max_questions=1)
    assert len(questions) >= 1
    item = questions[0]

    engine = TandemEngine()
    scorer = CandidateDifferentialScorer(engine=engine)

    candidates = item.metadata.get("all_candidates")
    assert candidates and len(candidates) >= 2, "Question metadata must provide all_candidates"
    assert item.ground_truth in candidates, (
        f"Precondition failed: gold answer '{item.ground_truth}' must be present in candidates"
    )

    # 1. Raw Likelihood Scoring
    res_raw = scorer.score_candidates(
        item.prompt,
        candidates,
        prefix_template="\n<answer>",
    )

    # 2. Contrastive PMI Scoring (unconditional neutral context prior subtracted)
    res_pmi = scorer.score_candidates(
        item.prompt,
        candidates,
        prefix_template="\n<answer>",
        neutral_context="The most likely diagnosis is:",
    )

    assert isinstance(res_raw, DifferentialScoringResult)
    assert len(res_raw.candidates) == len(candidates)
    assert res_raw.top_candidate.rank == 1

    # Verify every candidate has finite logprobs in [-20, 0]
    for c in res_raw.candidates:
        assert torch.isfinite(torch.tensor(c.token_logprobs)).all()
        assert not any(math.isnan(lp) for lp in c.token_logprobs)
        assert -20.0 <= c.mean_logprob <= 0.0

    for c in res_pmi.candidates:
        assert c.pmi_score is not None

    print("\n" + "=" * 65)
    print("REAL WEIGHTS 5-OPTION EVALUATION: RAW LIKELIHOOD")
    print("=" * 65)
    print(res_raw.summary())
    print("=" * 65)
    print("REAL WEIGHTS 5-OPTION EVALUATION: CONTRASTIVE PMI")
    print("=" * 65)
    print(res_pmi.summary())
    print("=" * 65)

