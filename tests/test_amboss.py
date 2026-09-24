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
            "rationale": "Klinefelter syndrome patients have lower incidence of prostate cancer.",
            "explanationWhy": "Klinefelter syndrome patients have lower incidence of prostate cancer.",
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
        "Prostate cancer": "lower incidence of prostate cancer in Klinefelter syndrome",
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
        "Prostate cancer": "lower incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    completion = (
        "<think>Prostate cancer ruled out because incidence is lower in Klinefelter syndrome.</think>"
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
        "Prostate cancer": "lower incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    completion = (
        "<think>"
        "1. Prostate cancer: ruled out because incidence is lower.\n"
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
        "Prostate cancer": "lower incidence of prostate cancer in Klinefelter syndrome",
        "Aortic dissection": "Marfan syndrome connective tissue disorder",
    }

    # 1 rule-out matched, gold failed
    comp_ro1 = (
        "<think>Prostate cancer is ruled out because incidence is lower in this syndrome.</think>"
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
        "Prostate cancer ruled out due to lower incidence. "
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
        "Prostate cancer": "lower incidence of prostate cancer in Klinefelter syndrome",
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
        "Option A ruled out without alpha beta. "
        "Option C ruled out unlike epsilon zeta. "
        "Option D ruled out lacks iota kappa. "
        "Option E ruled out, but omega sigma is absent. "
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
    buts = {"Prostate cancer": "lower incidence"}

    # Distractor name mentioned at start, keyword dumped at end >50 words away
    filler = " " + "word " * 60
    comp_stuffed = (
        f"<think>Prostate cancer was considered.{filler}However lower incidence occurs elsewhere.</think>"
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
    """Verify that answering with a generic subword ('Acute') does NOT match multi-word gold ('Acute cholecystitis'),
    while legitimate substantial clinical partials ('Cholecystitis') pass under the length-ratio guard (>= 0.50).
    """
    reward_fn = AmbossDifferentialReward()
    gt = "Acute cholecystitis"

    # Generic modifier ("Acute"): len 5 / 19 = 0.263 < 0.50 -> rejected
    comp_generic = "<think>High suspicion of acute condition.</think><answer>Acute</answer>"
    score_generic = reward_fn.compute_reward(prompt="case", completion=comp_generic, ground_truth=gt)
    assert score_generic == 0.0  # Must NOT match gold

    # Legitimate clinical partial ("Cholecystitis"): len 13 / 19 = 0.684 >= 0.50 -> accepted
    comp_partial = "<think>High suspicion.</think><answer>Cholecystitis</answer>"
    score_partial = reward_fn.compute_reward(prompt="case", completion=comp_partial, ground_truth=gt)
    assert score_partial == 2.0  # Legitimate clinical partial matches gold


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


def test_amboss_adversarial_affirmative_rationale_exploit_and_morphological_trap():
    """Verify that parroting affirmative rationale without contrast/negation earns zero credit,
    and morphological trap ('painless' containing substring 'pain') does not trigger false match.
    """
    reward_fn = AmbossDifferentialReward(gate_on_gold=False)
    gt = "Treponema pallidum"  # Syphilis: painless chancre
    candidates = ["Haemophilus ducreyi"]  # Chancroid: painful ulcers
    buts = {"Haemophilus ducreyi": "painful ulcers, fluctuant buboes"}

    # 1. Exploit: model parrots affirmative features with colon after candidate name without establishing patient lacks them
    comp_exploit = (
        "<think>Haemophilus ducreyi ruled out: painful ulcers, fluctuant buboes.</think>"
        "<answer>Treponema pallidum</answer>"
    )
    score_exploit = reward_fn.compute_reward(
        prompt="case", completion=comp_exploit, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    assert score_exploit == 2.0  # Only gold awarded; rule-out rejected due to missing contrast/negation

    # 2. Morphological trap: 'painless' contains substring 'pain'.
    # Word boundary matching ensures 'pain' does not match 'painless', and no false credit is given.
    comp_morph_trap = (
        "<think>Haemophilus ducreyi considered, but patient has a painless lesion.</think>"
        "<answer>Treponema pallidum</answer>"
    )
    score_morph = reward_fn.compute_reward(
        prompt="case", completion=comp_morph_trap, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    # Rationale has 'painful', completion has 'painless' -> regex \b rejects subword match
    assert score_morph == 2.0

    # 3. Legitimate clinical contrast: model explicitly contrasts chancroid's painful features or notes absence
    comp_legit = (
        "<think>Haemophilus ducreyi is unlikely because unlike chancroid which presents with painful ulcers, "
        "this patient has no buboes and the ulcer is painless.</think>"
        "<answer>Treponema pallidum</answer>"
    )
    score_legit = reward_fn.compute_reward(
        prompt="case", completion=comp_legit, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    # Gold (2.0) + 1 verified ruleout (0.20) = 2.20
    assert score_legit == pytest.approx(2.20, abs=1e-5)


def test_amboss_adversarial_clinical_subword_false_positives():
    """Verify that clinical subwords do NOT match different clinical entities:
    - 'Carditis' must NOT match 'Myocarditis' or 'Pericarditis'
    - 'Cystitis' must NOT match 'Cholecystitis'
    - 'Pain' must NOT match 'Painless'
    """
    reward_fn = AmbossDifferentialReward()

    # 1. Carditis vs Myocarditis / Pericarditis
    comp_carditis = "<think>Exam suggests cardiac issue.</think><answer>Carditis</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_carditis, ground_truth="Myocarditis") == 0.0
    assert reward_fn.compute_reward(prompt="case", completion=comp_carditis, ground_truth="Pericarditis") == 0.0

    # 2. Cystitis (bladder) vs Cholecystitis (gallbladder)
    comp_cystitis = "<think>Abdominal inflammation.</think><answer>Cystitis</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_cystitis, ground_truth="Cholecystitis") == 0.0
    assert reward_fn.compute_reward(prompt="case", completion=comp_cystitis, ground_truth="Acute cholecystitis") == 0.0

    # 3. Pain vs Painless (forward match with word boundaries)
    comp_painless = "<think>Genital examination.</think><answer>Painless</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_painless, ground_truth="Pain") == 0.0


def test_amboss_adversarial_distractor_rationale_containing_contrast_words_parrot_exploit():
    """Verify that when the AMBOSS distractor rationale itself contains contrast words like 'unlike' or 'no',
    a model that merely parrots the distractor's affirmative pathology receives zero rule-out credit.
    """
    reward_fn = AmbossDifferentialReward(gate_on_gold=False)
    gt = "Treponema pallidum"
    candidates = ["Haemophilus ducreyi"]
    # AMBOSS rationale contains "unlike" and "no", but the distractor's pathology is "painful ulcers"
    buts = {
        "Haemophilus ducreyi": "unlike syphilis which causes no pain, chancroid presents with painful ulcers and fluctuant buboes"
    }

    # Model parrots the distractor's affirmative pathology without establishing patient-specific contrast
    comp_parrot = (
        "<think>Haemophilus ducreyi ruled out: painful ulcers, fluctuant buboes.</think>"
        "<answer>Treponema pallidum</answer>"
    )
    score = reward_fn.compute_reward(
        prompt="case", completion=comp_parrot, ground_truth=gt,
        differential_candidates=candidates, distractor_buts=buts,
    )
    # The AMBOSS rationale had 'unlike' and 'no', but the model's text lacks contrast -> zero rule-out credit
    assert score == 2.0  # Gold awarded (2.0), rule-out credit rejected (0.0)


def test_amboss_diagnosis_stop_words_and_clinical_head_nouns():
    """Verify diagnosis stop-word preservation and clinical head noun distinctions:
    - Cushing disease vs Cushing syndrome -> reject (score 0.0)
    - Addison disease vs Addisonian crisis -> reject (score 0.0)
    - nephrotic syndrome vs nephritic syndrome -> reject (score 0.0)
    - Generic head nouns alone ('Syndrome', 'Disease') -> reject (score 0.0)
    - Compound partials ('Cholecystitis' for 'Acute cholecystitis') -> accept (score 2.0)
    """
    reward_fn = AmbossDifferentialReward()

    # 1. Cushing disease vs Cushing syndrome
    comp_cushing_syn = "<think>Elevated ACTH and cortisol.</think><answer>Cushing syndrome</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_cushing_syn, ground_truth="Cushing disease") == 0.0

    comp_cushing_dis = "<think>Elevated ACTH and cortisol.</think><answer>Cushing disease</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_cushing_dis, ground_truth="Cushing syndrome") == 0.0

    # 2. Addison disease vs Addisonian crisis
    comp_addison_crisis = "<think>Adrenal insufficiency presentation.</think><answer>Addisonian crisis</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_addison_crisis, ground_truth="Addison disease") == 0.0

    comp_addison_dis = "<think>Adrenal insufficiency presentation.</think><answer>Addison disease</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_addison_dis, ground_truth="Addisonian crisis") == 0.0

    # 3. Nephrotic syndrome vs Nephritic syndrome
    comp_nephritic = "<think>Renal findings with hematuria.</think><answer>Nephritic syndrome</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_nephritic, ground_truth="Nephrotic syndrome") == 0.0

    comp_nephrotic = "<think>Heavy proteinuria and edema.</think><answer>Nephrotic syndrome</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_nephrotic, ground_truth="Nephritic syndrome") == 0.0

    # 4. Generic head nouns alone must NOT match specific diagnoses
    comp_syndrome = "<think>Complex clinical findings.</think><answer>Syndrome</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_syndrome, ground_truth="Cushing syndrome") == 0.0

    comp_disease = "<think>Infectious disease etiology.</think><answer>Disease</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_disease, ground_truth="Graves disease") == 0.0

    # 5. Legitimate compound partials must still match
    comp_cholecystitis = "<think>RUQ pain and Murphy sign.</think><answer>Cholecystitis</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_cholecystitis, ground_truth="Acute cholecystitis") == 2.0

    comp_appendicitis = "<think>RLQ pain and McBurney tenderness.</think><answer>Appendicitis</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_appendicitis, ground_truth="Acute appendicitis") == 2.0


def test_amboss_short_clinical_tokens_allowlist():
    """Verify that short 2-character clinical tokens (pH, pO2, BP, etc.) are preserved
    and can trigger rule-out credit when valid contrast is established.
    """
    from tandem.rl.amboss import _extract_keywords, SHORT_CLINICAL_ALLOWLIST

    # Test keyword extraction preserves short tokens
    text = "Arterial blood gas shows abnormal pH and low pO2 with elevated BP"
    kws = _extract_keywords(text, min_len=4)
    for token in ["ph", "po2", "bp"]:
        assert token in kws, f"Expected short token '{token}' in extracted keywords"

    # Test ruleout credit using short clinical token
    reward_fn = AmbossDifferentialReward(gate_on_gold=False)
    gt = "Metabolic acidosis"
    candidates = ["Respiratory acidosis"]
    buts = {
        "Respiratory acidosis": "unlike metabolic acidosis which shows low pH with low bicarbonate, respiratory acidosis shows high pCO2"
    }

    comp = (
        "<think>Respiratory acidosis is ruled out because patient lacks elevated pco2.</think>"
        "<answer>Metabolic acidosis</answer>"
    )
    score = reward_fn.compute_reward(
        prompt="case",
        completion=comp,
        ground_truth=gt,
        differential_candidates=candidates,
        distractor_buts=buts,
    )
    # Gold (2.0) + ruleout credit for respiratory acidosis (0.20) = 2.20
    assert abs(score - 2.20) < 1e-4


def test_amboss_diagnosis_abbreviation_and_possessive_controls():
    """Verify abbreviation expansion and possessive normalization in gold matching."""
    reward_fn = AmbossDifferentialReward()

    comp_s = "<think>Gram-positive diplococci.</think><answer>S. pneumoniae</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_s, ground_truth="Streptococcus pneumoniae") == 2.0

    comp_poss = "<think>Elevated ACTH and cortisol.</think><answer>Cushing's syndrome</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_poss, ground_truth="Cushing syndrome") == 2.0


@pytest.mark.xfail(strict=True, reason="pneumococcus synonym requires CUI/alias layer")
def test_amboss_pneumococcus_synonym_known_gap():
    """Known recall gap: pneumococcus is a true synonym of S. pneumoniae but not a token variant."""
    reward_fn = AmbossDifferentialReward()
    comp = "<think>Gram-positive diplococci.</think><answer>pneumococcus</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp, ground_truth="Streptococcus pneumoniae") == 2.0


def test_amboss_modifier_aware_partial_matching_controls():
    """Verify clinical modifiers can be dropped/added, but non-modifier token changes reject."""
    reward_fn = AmbossDifferentialReward()

    # Modifier-only variation: accept both directions.
    comp_partial = "<think>RUQ pain and Murphy sign.</think><answer>Cholecystitis</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_partial, ground_truth="Acute cholecystitis") == 2.0

    comp_full = "<think>RUQ pain and Murphy sign.</think><answer>Acute cholecystitis</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_full, ground_truth="Cholecystitis") == 2.0

    # Non-modifier differences: reject.
    comp_vit_c = "<think>Vitamin deficiency.</think><answer>Vitamin C</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_vit_c, ground_truth="Vitamin A") == 0.0

    comp_chlor = "<think>Hypertension regimen.</think><answer>Chlorthalidone</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_chlor, ground_truth="Lisinopril and chlorthalidone") == 0.0


def test_amboss_abbreviation_ambiguity_and_cased_short_tokens():
    """Verify S. pneumoniae expansion is specific, and the short-token allowlist is case-cased."""
    from tandem.rl.amboss import _extract_keywords

    reward_fn = AmbossDifferentialReward()

    comp_pneumo = "<think>Gram-positive diplococci.</think><answer>S. pneumoniae</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_pneumo, ground_truth="Streptococcus pneumoniae") == 2.0

    # S. aureus is Staphylococcus, not Streptococcus; it must not be expanded as strep.
    comp_aureus = "<think>Gram-positive cocci in clusters.</think><answer>S. aureus</answer>"
    assert reward_fn.compute_reward(prompt="case", completion=comp_aureus, ground_truth="Streptococcus pneumoniae") == 0.0

    # Cased allowlist: pH is preserved, lowercase ph is not a short-token bypass.
    assert "ph" in _extract_keywords("pH", min_len=4)
    assert "ph" not in _extract_keywords("ph", min_len=4)


def test_amboss_order_sensitive_false_positives():
    """Verify that identical token sets in different order are not treated as equivalent."""
    reward_fn = AmbossDifferentialReward()
    pairs = [
        (
            "Left-to-right shunt through the ventricular septum",
            "Right-to-left shunt through the ventricular septum",
        ),
        (
            "Increased specificity and decreased negative predictive value",
            "Decreased specificity and increased negative predictive value",
        ),
        (
            "Stop playing soccer, continue strength training, and do not buy a ski pass",
            "Continue playing soccer, stop strength training, and do not buy a ski pass",
        ),
    ]
    for gold, cand in pairs:
        comp = f"<think>reasoning</think><answer>{cand}</answer>"
        assert reward_fn.compute_reward(prompt="case", completion=comp, ground_truth=gold) == 0.0
