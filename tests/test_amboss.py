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
    # Under magnitude dominance: Gold alone = 2.0
    assert score == 2.0


def test_amboss_differential_reward_gold_plus_one_ruleout():
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"
    candidates = ["Prostate cancer", "Aortic dissection"]
    buts = {
        "Prostate cancer": "decreased incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    completion = (
        "<think>Prostate cancer ruled out because incidence is decreased in Klinefelter syndrome.</think>"
        "<answer>Breast cancer</answer>"
    )
    score = reward_fn.compute_reward(
        prompt="prompt",
        completion=completion,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    # Gold (2.0) + 1 rule-out (0.20) = 2.20
    assert score == pytest.approx(2.20, abs=1e-5)


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
    # Gold (2.0) + 2 rule-outs (0.40) = 2.40
    assert score == pytest.approx(2.40, abs=1e-5)


def test_amboss_differential_reward_matching_ruleouts_failing_gold():
    # Under magnitude dominance (default gate_on_gold=False):
    # Rule-outs keep advantage variance alive without ever outscoring a right answer
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"
    candidates = ["Prostate cancer", "Aortic dissection"]
    buts = {
        "Prostate cancer": "decreased incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    # 1 rule-out matched, gold failed
    comp_ro1 = (
        "<think>Prostate cancer is ruled out because incidence is decreased in this syndrome.</think>"
        "<answer>Aortic dissection</answer>"
    )
    score_ro1 = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_ro1,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    # Partial credit = 0.20 (strictly < 2.00)
    assert score_ro1 == pytest.approx(0.20, abs=1e-5)

    # 2 rule-outs matched, gold failed
    comp_ro2 = (
        "<think>"
        "Prostate cancer ruled out due to decreased incidence. "
        "Aortic dissection ruled out because connective tissue disorder of Marfan syndrome is absent."
        "</think>"
        "<answer>Leukemia</answer>"
    )
    score_ro2 = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_ro2,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    # 2 rule-outs = 0.40. MAGNITUDE DOMINANCE: 0.40 is far below min correct (2.00)
    assert score_ro2 == pytest.approx(0.40, abs=1e-5)
    assert score_ro2 < reward_fn.gold_reward

    # With gate_on_gold=True explicitly set: gives 0.0
    gated_reward = AmbossDifferentialReward(gate_on_gold=True)
    assert gated_reward.compute_reward(
        prompt="prompt", completion=comp_ro2, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts
    ) == 0.0


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
    # Gold (2.0) + 4 rule-outs (4 * 0.20 = 0.80) = 2.80 (never clipped prematurely)
    assert score == pytest.approx(2.80, abs=1e-5)
    assert score <= 3.0



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
    assert score_letter == 2.0

    # Letter + content in answer tag
    comp_both = "<think>Reasoning...</think><answer>B. Breast cancer</answer>"
    score_both = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_both,
        ground_truth="Breast cancer",
        options=options,
    )
    assert score_both == 2.0


def test_amboss_differential_reward_dropped_hypothesis_not_scored():
    """Verify that merely mentioning the gold diagnosis in <think> without confirming
    it in <answer> or listing it in <differential> does NOT score gold credit.
    """
    reward_fn = AmbossDifferentialReward()
    gt = "Breast cancer"

    # Mentioned in think as a rejected hypothesis, dropped for Leukemia in answer
    comp_dropped = (
        "<think>We must consider breast cancer, but patient presentation favors leukemia.</think>"
        "<answer>Leukemia</answer>"
    )
    score = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_dropped,
        ground_truth=gt,
    )
    assert score == 0.0, "Dropped hypothesis in <think> must not collect gold credit"


def test_amboss_differential_reward_no_asymmetric_prefix_overcredit():
    """Verify that an incomplete prefix like 'Acute' does NOT match 'Acute cholecystitis'."""
    reward_fn = AmbossDifferentialReward()
    gt = "Acute cholecystitis"

    # Incomplete single-word prefix in <answer>
    comp_prefix = "<think>Reasoning...</think><answer>Acute</answer>"
    score = reward_fn.compute_reward(
        prompt="prompt",
        completion=comp_prefix,
        ground_truth=gt,
    )
    assert score == 0.0, "Incomplete single-word prefix must not match multi-word disease"


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
    assert score == 2.0


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
    # Gold (2.0) + 1 rule-out (0.20) = 2.20
    assert score_correct == pytest.approx(2.20, abs=1e-5)

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
    assert score_affirmed == 2.0  # Only gold credit (2.0), rule-out rejected due to negation violation


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
    assert score_stuffed == 2.0  # Gold awarded (2.0), distractor rule-out rejected due to proximity window breach


@pytest.mark.skipif(not AMBOSS_QUESTIONS_DIR.is_dir(), reason="Amboss questions directory not accessible")
def test_amboss_deterministic_train_eval_split():
    """Verify that train and eval splits are deterministic, disjoint, and non-empty."""
    train_items = load_amboss_questions(AMBOSS_QUESTIONS_DIR, max_questions=50, split="train", eval_split_ratio=0.20)
    eval_items = load_amboss_questions(AMBOSS_QUESTIONS_DIR, max_questions=50, split="eval", eval_split_ratio=0.20)

    train_qids = {item.metadata["qid"] for item in train_items}
    eval_qids = {item.metadata["qid"] for item in eval_items}

    # Splits must be disjoint
    assert train_qids.isdisjoint(eval_qids), "Train and eval question sets must be mutually exclusive"
    assert len(train_items) > 0, "Train split must be non-empty"
    assert len(eval_items) > 0, "Eval split must be non-empty"


# =====================================================================
# 7. Adversarial Red-Team Reward Suite
# =====================================================================

def test_amboss_adversarial_affirmation_verbs():
    """Verify that expanded affirmative verbs ('reveals', 'demonstrates', 'notable')
    invalidate distractor rule-out credit when rationale asserts absence.
    """
    reward_fn = AmbossDifferentialReward(gate_on_gold=False)
    gt = "Klebsiella granulomatis"
    candidates = ["Treponema pallidum"]
    buts = {"Treponema pallidum": "painless chancre without prominent lymphadenopathy"}

    # Test verb: 'reveals'
    comp_reveals = (
        "<think>Treponema pallidum was considered, but physical examination reveals prominent lymphadenopathy.</think>"
        "<answer>Klebsiella granulomatis</answer>"
    )
    score_reveals = reward_fn.compute_reward(
        prompt="case", completion=comp_reveals, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    assert score_reveals == 2.0  # Only gold awarded; rule-out rejected due to affirmation breach

    # Test verb: 'demonstrates'
    comp_demonstrates = (
        "<think>Treponema pallidum evaluation demonstrates severe lymphadenopathy.</think>"
        "<answer>Klebsiella granulomatis</answer>"
    )
    score_demonstrates = reward_fn.compute_reward(
        prompt="case", completion=comp_demonstrates, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    assert score_demonstrates == 2.0

    # Test phrase: 'notable'
    comp_notable = (
        "<think>Treponema pallidum finding notable lymphadenopathy on groin exam.</think>"
        "<answer>Klebsiella granulomatis</answer>"
    )
    score_notable = reward_fn.compute_reward(
        prompt="case", completion=comp_notable, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    assert score_notable == 2.0


def test_amboss_adversarial_keyword_stuffing_without_candidate():
    """Verify that dumping rationale keywords without mentioning the candidate distractor earns zero rule-out credit."""
    reward_fn = AmbossDifferentialReward(gate_on_gold=False)
    gt = "Acute cholecystitis"
    candidates = ["Acute appendicitis"]
    buts = {"Acute appendicitis": "periumbilical pain migrating to McBurney point"}

    # Completion dumps keywords ('migrating', 'mcburney') without ever naming 'appendicitis'
    comp_stuffed = (
        "<think>The examination shows migrating discomfort near mcburney area without clear cause.</think>"
        "<answer>Acute cholecystitis</answer>"
    )
    score = reward_fn.compute_reward(
        prompt="case", completion=comp_stuffed, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    # Distractor name was never mentioned, so proximity attribution rejects it
    assert score == 2.0  # Gold credit only


def test_amboss_adversarial_asymmetric_substring_gold_rejection():
    """Verify that answering with a generic subword ('Acute') does NOT match multi-word gold ('Acute cholecystitis')."""
    reward_fn = AmbossDifferentialReward()
    gt = "Acute cholecystitis"

    comp_generic = "<think>High suspicion of acute condition.</think><answer>Acute</answer>"
    score_generic = reward_fn.compute_reward(prompt="case", completion=comp_generic, ground_truth=gt)
    assert score_generic == 0.0  # Must NOT match gold

    comp_inverted = "<think>High suspicion.</think><answer>Cholecystitis</answer>"
    score_inverted = reward_fn.compute_reward(prompt="case", completion=comp_inverted, ground_truth=gt)
    assert score_inverted == 0.0  # Partial single-word does NOT match compound gold


def test_amboss_adversarial_rule_out_without_gold_magnitude_dominance():
    """Verify that articulate-but-wrong reasoning with 4 valid rule-outs scores strictly <= 0.80,
    guaranteeing strict mathematical dominance over any gold-correct answer (>= 2.00).
    """
    reward_fn = AmbossDifferentialReward(
        gold_reward=2.0,
        ruleout_credit_per_candidate=0.20,
        max_ruleout_credit=0.80,
        gate_on_gold=False,
    )
    gt = "Streptococcus pneumoniae"
    candidates = [
        "Salmonella paratyphi",
        "Nontypeable Haemophilus influenzae",
        "Neisseria meningitidis",
        "Staphylococcus aureus",
    ]
    buts = {
        "Salmonella paratyphi": "osteomyelitis risk in bone crisis",
        "Nontypeable Haemophilus influenzae": "unencapsulated otitis media pathogen",
        "Neisseria meningitidis": "petechial purpuric rash presentation",
        "Staphylococcus aureus": "cavitary abscess formation on chest radiograph",
    }

    # Wrong answer given ("Mycoplasma"), but all 4 distractors correctly analyzed in <rule_out> blocks
    comp_articulate_wrong = (
        "<think>"
        '<rule_out target="Salmonella paratyphi">osteomyelitis risk in bone crisis is absent</rule_out>'
        '<rule_out target="Nontypeable Haemophilus influenzae">unencapsulated otitis media pathogen is not seen</rule_out>'
        '<rule_out target="Neisseria meningitidis">petechial purpuric rash presentation not present</rule_out>'
        '<rule_out target="Staphylococcus aureus">cavitary abscess formation on chest radiograph absent</rule_out>'
        "</think>"
        "<answer>Mycoplasma pneumoniae</answer>"
    )

    score_wrong = reward_fn.compute_reward(
        prompt="case", completion=comp_articulate_wrong, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    assert score_wrong == 0.80  # Exactly max ruleout credit

    # Correct answer with ZERO ruleouts
    comp_correct_bare = "<answer>Streptococcus pneumoniae</answer>"
    score_correct = reward_fn.compute_reward(
        prompt="case", completion=comp_correct_bare, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    assert score_correct == 2.00

    # Strict magnitude dominance invariant: wrong answer max <= 0.80 < 2.00 <= right answer min
    assert score_wrong <= 0.80
    assert score_correct >= 2.00
    assert score_correct > score_wrong
    assert (score_correct - score_wrong) >= 1.20  # Minimum 1.20 gap guarantees no score inversion


