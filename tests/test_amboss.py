from __future__ import annotations

import json
from pathlib import Path
import pytest

from tandem.rl.amboss import (
    AmbossDifferentialReward,
    AmbossQuestion,
    clean_amboss_html,
    load_amboss_questions,
    parse_amboss_json,
)
from tandem.rl.dataset import PromptItem

AMBOSS_QUESTIONS_DIR = Path(
    "/Users/rohanmaster/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Resources/data/raw/amboss_qbank/questions"
)


# =====================================================================
# 1. HTML Cleaning Tests
# =====================================================================

def test_clean_amboss_html_styles_and_scripts():
    raw_html = (
        "<style>\n.nowrap { white-space: nowrap; }\n.btn { color: red; }\n</style>"
        "<p>Patient presented with <span class=\"Highlight\">fever</span>.</p>"
        "<script>console.log('test');</script>"
    )
    cleaned = clean_amboss_html(raw_html)
    assert "<style>" not in cleaned
    assert "white-space" not in cleaned
    assert "<script>" not in cleaned
    assert "console.log" not in cleaned
    assert cleaned == "Patient presented with fever."


def test_clean_amboss_html_block_and_inline_tags():
    raw_html = (
        "<p>First paragraph.</p>"
        "<p>Second paragraph with <b>bold</b> and <i>italic</i> words and a <br/>line break.</p>"
        "<div>Div block with <span class=\"nowrap\">94<sup>th</sup> percentile</span>.</div>"
    )
    cleaned = clean_amboss_html(raw_html)
    assert cleaned == (
        "First paragraph. Second paragraph with bold and italic words and a line break. "
        "Div block with 94th percentile."
    )


def test_clean_amboss_html_entities_and_whitespace():
    raw_html = (
        "<p>Blood pressure &gt; 140/90&nbsp;mm&nbsp;Hg &amp; pulse was &lt; 50/min.&#39;</p>\n\n"
        "   \t  <p>Status:  stable.  </p>"
    )
    cleaned = clean_amboss_html(raw_html)
    assert cleaned == "Blood pressure > 140/90 mm Hg & pulse was < 50/min.' Status: stable."


def test_clean_amboss_html_edge_cases():
    assert clean_amboss_html("") == ""
    assert clean_amboss_html(None) == ""
    assert clean_amboss_html("   \n\t  ") == ""
    assert clean_amboss_html("Clean plain text without tags") == "Clean plain text without tags"


# =====================================================================
# 2. AmbossQuestion Dataclass Tests
# =====================================================================

def test_amboss_question_dataclass():
    distractors = [
        {
            "content": "Prostate cancer",
            "letter": "a",
            "rationale": "Klinefelter syndrome patients have decreased incidence of prostate cancer.",
            "explanationWhy": "Klinefelter syndrome patients have decreased incidence of prostate cancer.",
            "explanationBut": "",
        },
        {
            "content": "Aortic dissection",
            "letter": "c",
            "rationale": "Marfan syndrome connective tissue disorder.",
            "explanationWhy": "Marfan syndrome connective tissue disorder.",
            "explanationBut": "",
        },
    ]
    options = [
        {"content": "Prostate cancer", "letter": "a", "isCorrect": False},
        {"content": "Breast cancer", "letter": "b", "isCorrect": True},
        {"content": "Aortic dissection", "letter": "c", "isCorrect": False},
    ]

    q = AmbossQuestion(
        qid="test_001",
        vignette="A 16-year-old boy presents with gynecomastia and tall stature.",
        learning_objective="Recognize Klinefelter syndrome and cancer risks.",
        gold_answer="Breast cancer",
        distractors=distractors,
        options=options,
        gold_why="Testicular hypoplasia leads to elevated estrogen and breast cancer.",
    )

    assert q.qid == "test_001"
    assert q.differential_candidates == ["Prostate cancer", "Aortic dissection"]
    assert "Prostate cancer" in q.distractor_buts
    assert "Aortic dissection" in q.distractor_buts

    item = q.to_prompt_item(include_options=True)
    assert isinstance(item, PromptItem)
    assert "A 16-year-old boy presents" in item.prompt
    assert "Options:" in item.prompt
    assert "A. Prostate cancer" in item.prompt
    assert "B. Breast cancer" in item.prompt
    assert "<think>" in item.prompt
    assert "<answer>" in item.prompt
    assert item.ground_truth == "Breast cancer"
    assert item.metadata["differential_candidates"] == ["Prostate cancer", "Aortic dissection"]
    assert item.metadata["learning_objective"] == "Recognize Klinefelter syndrome and cancer risks."
    assert "elevated estrogen" in item.metadata["gold_why"]
    assert item.metadata["distractor_buts"]["Prostate cancer"] == distractors[0]["rationale"]


# =====================================================================
# 3. Ingestion Tests on Actual Amboss Directory
# =====================================================================

@pytest.mark.skipif(not AMBOSS_QUESTIONS_DIR.is_dir(), reason="Amboss questions directory not accessible")
def test_load_amboss_questions_sample():
    items = load_amboss_questions(AMBOSS_QUESTIONS_DIR, max_questions=5)
    assert len(items) == 5

    for item in items:
        assert isinstance(item, PromptItem)
        assert len(item.prompt) > 0
        assert "<think>" in item.prompt
        assert "<answer>" in item.prompt
        assert item.ground_truth is not None and len(item.ground_truth) > 0

        # Required metadata fields
        assert "differential_candidates" in item.metadata
        assert "learning_objective" in item.metadata
        assert "gold_why" in item.metadata
        assert "distractor_buts" in item.metadata

        candidates = item.metadata["differential_candidates"]
        assert isinstance(candidates, list)
        assert len(candidates) > 0

        distractor_buts = item.metadata["distractor_buts"]
        assert isinstance(distractor_buts, dict)
        assert len(distractor_buts) == len(candidates)


# =====================================================================
# 4. AmbossDifferentialReward Synthetic Completion Tests
# =====================================================================

def test_amboss_differential_reward_gold_only():
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"
    candidates = ["Prostate cancer", "Aortic dissection"]
    buts = {
        "Prostate cancer": "decreased incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    completion = "<think>Patient has gynecomastia and small testes.</think><answer>Breast cancer</answer>"
    score = reward_fn.compute_reward(
        prompt="prompt",
        completion=completion,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    assert score == 1.0


def test_amboss_differential_reward_gold_plus_one_ruleout():
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"
    candidates = ["Prostate cancer", "Aortic dissection"]
    buts = {
        "Prostate cancer": "decreased incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    completion = (
        "<think>Rule out prostate cancer because incidence is decreased in Klinefelter syndrome.</think>"
        "<answer>Breast cancer</answer>"
    )
    score = reward_fn.compute_reward(
        prompt="prompt",
        completion=completion,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    # Gold (1.0) + 1 rule-out (0.25) = 1.25
    assert score == 1.25


def test_amboss_differential_reward_gold_plus_two_ruleouts():
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"
    candidates = ["Prostate cancer", "Aortic dissection"]
    buts = {
        "Prostate cancer": "decreased incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    completion = (
        "<think>"
        "1. Prostate cancer: ruled out because incidence is decreased.\n"
        "2. Aortic dissection: characteristic of Marfan syndrome with connective tissue disorder, "
        "not seen in this patient.\n"
        "</think>"
        "<answer>Breast cancer</answer>"
    )
    score = reward_fn.compute_reward(
        prompt="prompt",
        completion=completion,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    # Gold (1.0) + 2 rule-outs (0.50) = 1.50
    assert score == 1.50


def test_amboss_differential_reward_matching_ruleouts_failing_gold():
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"
    candidates = ["Prostate cancer", "Aortic dissection"]
    buts = {
        "Prostate cancer": "decreased incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    # 1 rule-out matched, gold failed
    comp_ro1 = (
        "<think>Rule out prostate cancer because incidence is decreased in this syndrome.</think>"
        "<answer>Aortic dissection</answer>"
    )
    # With default gate_on_gold=True: articulate wrong diagnosis gets 0.0 (exploit blocked)
    score_gated = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_ro1,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    assert score_gated == 0.0, "Rule-out credit must be gated on gold diagnosis match"

    # With gate_on_gold=False explicitly enabled: awards partial credit
    ungated_reward = AmbossDifferentialReward(gate_on_gold=False)
    score_ro1 = ungated_reward.compute_reward(
        prompt="prompt",
        completion=comp_ro1,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    assert score_ro1 == 0.25

    # 2 rule-outs matched, gold failed, ungated
    comp_ro2 = (
        "<think>"
        "Rule out prostate cancer due to decreased incidence. "
        "Rule out aortic dissection because connective tissue disorder of Marfan syndrome is absent."
        "</think>"
        "<answer>Leukemia</answer>"
    )
    score_ro2 = ungated_reward.compute_reward(
        prompt="prompt",
        completion=comp_ro2,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    assert score_ro2 == 0.50


def test_amboss_differential_reward_failing_both():
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"
    candidates = ["Prostate cancer", "Aortic dissection"]
    buts = {
        "Prostate cancer": "decreased incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    comp_fail = "<think>I am not sure about this diagnosis.</think><answer>Unknown</answer>"
    score_fail = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_fail,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    assert score_fail == 0.0


def test_amboss_differential_reward_max_score_cap():
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"
    candidates = ["Option A", "Option C", "Option D", "Option E"]
    buts = {
        "Option A": "alpha beta gamma delta",
        "Option C": "epsilon zeta eta theta",
        "Option D": "iota kappa lambda sigma",
        "Option E": "omega sigma theta phi",
    }

    completion = (
        "<think>"
        "Option A ruled out with alpha beta. "
        "Option C ruled out with epsilon zeta. "
        "Option D ruled out with iota kappa. "
        "Option E ruled out with omega sigma. "
        "</think>"
        "<answer>Breast cancer</answer>"
    )
    score = reward_fn.compute_reward(
        prompt="prompt",
        completion=completion,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    # Gold (1.0) + 4 rule-outs (4 * 0.25 = 1.0) = 2.0 (maximum)
    assert score == 2.0
    assert 0.0 <= score <= 2.0



def test_amboss_differential_reward_option_letter_matching():
    reward_fn = AmbossDifferentialReward()
    options = [
        {"content": "Prostate cancer", "letter": "a", "isCorrect": False},
        {"content": "Breast cancer", "letter": "b", "isCorrect": True},
    ]

    # Letter in answer tag
    comp_letter = "<think>Reasoning...</think><answer>B</answer>"
    score_letter = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_letter,
        ground_truth="Breast cancer",
        options=options,
    )
    assert score_letter == 1.0

    # Letter + content in answer tag
    comp_both = "<think>Reasoning...</think><answer>B. Breast cancer</answer>"
    score_both = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_both,
        ground_truth="Breast cancer",
        options=options,
    )
    assert score_both == 1.0


def test_amboss_differential_reward_no_answer_tag_fallback():
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"

    comp_untagged = "Clinical reasoning suggests the primary concern is breast cancer due to elevated estrogen."
    score = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_untagged,
        ground_truth=gt,
    )
    assert score == 1.0


# =====================================================================
# 5. Integration Test with Loaded Real Question
# =====================================================================

@pytest.mark.skipif(not AMBOSS_QUESTIONS_DIR.is_dir(), reason="Amboss questions directory not accessible")
def test_amboss_differential_reward_with_real_question():
    items = load_amboss_questions(AMBOSS_QUESTIONS_DIR, max_questions=1)
    assert len(items) == 1
    item = items[0]

    reward_fn = AmbossDifferentialReward()

    # 1. Matching gold answer
    gold_comp = f"<think>Reasoning...</think><answer>{item.ground_truth}</answer>"
    gold_score = reward_fn(
        prompt=item.prompt,
        completion=gold_comp,
        ground_truth=item.ground_truth,
        **item.metadata,
    )
    assert gold_score >= 1.0

    # 2. Failing answer
    fail_comp = "<think>I cannot solve this case.</think><answer>Completely wrong disease</answer>"
    fail_score = reward_fn(
        prompt=item.prompt,
        completion=fail_comp,
        ground_truth=item.ground_truth,
        **item.metadata,
    )
    assert fail_score == 0.0


def test_amboss_differential_tag_recall():
    """Verify that gold diagnosis present in <differential> candidate block scores 1.0 even without <answer> tag."""
    reward_fn = AmbossDifferentialReward()
    gt = "Streptococcus pneumoniae"

    comp = (
        "<think>Considering acute pulmonary infection in SCD.</think>"
        "<differential>"
        "<candidate>Streptococcus pneumoniae</candidate>"
        "<candidate>Salmonella paratyphi</candidate>"
        "</differential>"
    )
    score = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp,
        ground_truth=gt,
    )
    assert score == 1.0


def test_amboss_negation_scope_check():
    """Verify that when a distractor rationale relies on absent findings ('no lymphadenopathy'),
    affirming presence ('shows prominent lymphadenopathy') invalidates the rule-out credit.
    """
    reward_fn = AmbossDifferentialReward(gate_on_gold=False)
    gt = "Klebsiella granulomatis"
    candidates = ["Treponema pallidum"]
    buts = {"Treponema pallidum": "causes painless chancre with prominent inguinal lymphadenopathy, whereas patient has no lymphadenopathy"}

    # Case A: Correctly recognizing absence -> credit awarded
    comp_correct = (
        "<think>Treponema pallidum is ruled out because patient has no lymphadenopathy.</think>"
        "<answer>Klebsiella granulomatis</answer>"
    )
    score_correct = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_correct,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    assert score_correct == 1.25

    # Case B: False affirmation ("severe lymphadenopathy present") -> negation violated, credit rejected
    comp_affirmed = (
        "<think>Treponema pallidum because examination shows prominent lymphadenopathy.</think>"
        "<answer>Klebsiella granulomatis</answer>"
    )
    score_affirmed = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_affirmed,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    assert score_affirmed == 1.0  # Only gold credit, rule-out rejected due to negation violation


def test_amboss_proximity_attribution():
    """Verify that keyword stuffing 100 words away from the distractor candidate does not trigger rule-out credit."""
    reward_fn = AmbossDifferentialReward(gate_on_gold=False)
    gt = "Breast cancer"
    candidates = ["Prostate cancer"]
    buts = {"Prostate cancer": "decreased incidence"}

    # Distractor name mentioned at start, keyword dumped at end >50 words away
    filler = " " + "word " * 60
    comp_stuffed = (
        f"<think>Prostate cancer was considered.{filler}However decreased incidence occurs elsewhere.</think>"
        "<answer>Breast cancer</answer>"
    )
    score_stuffed = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_stuffed,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    assert score_stuffed == 1.0  # Gold awarded, distractor rule-out rejected due to proximity window breach
