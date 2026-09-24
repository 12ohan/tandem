from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from tandem.rl.dataset import PromptItem
from tandem.rl.reward import BaseReward

# Common English and clinical generic stop words to exclude during keyword extraction
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "almost", "also", "although",
    "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "let's", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "won't", "would", "you", "your",
    "yours", "yourself", "yourselves", "patient", "patients", "symptoms", "findings", "finding",
    "presentation", "present", "presents", "presented", "case", "cases", "history", "shown",
    "seen", "result", "results", "likely", "unlikely", "common", "typically", "associated",
    "syndrome", "syndromes", "disease", "diseases", "disorder", "disorders", "condition", "conditions",
}


def clean_amboss_html(html_str: Optional[str]) -> str:
    """Robust regex-based cleaner for Amboss HTML content.

    Removes <style> and <script> blocks, handles breaks/blocks cleanly,
    strips all HTML tags, unescapes entities, and collapses whitespace.
    """
    if not html_str or not isinstance(html_str, str):
        return ""

    # 1. Remove style and script tags with their inner contents
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", html_str)
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)

    # 2. Convert block-level tags and line breaks into spaces to avoid accidental word collisions
    text = re.sub(r"(?i)<(?:br|p|div|tr|li|h[1-6]|blockquote)\b[^>]*>", " ", text)
    text = re.sub(r"(?i)</(?:p|div|tr|li|h[1-6]|blockquote)>", " ", text)

    # 3. Remove all remaining tags (inline tags: <span>, <a>, <b>, <i>, <sup>, etc.)
    text = re.sub(r"<[^>]+>", "", text)

    # 4. Unescape HTML entities (e.g., &amp;, &lt;, &gt;, &nbsp;, etc.)
    text = html.unescape(text)

    # 5. Normalize non-breaking and special spaces
    text = text.replace("\xa0", " ")

    # 6. Collapse consecutive whitespace into single spaces and strip ends
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_text(text: str) -> str:
    """Normalize text for clinical string matching (lowercased, alphanumeric tokens)."""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_keywords(
    text: str,
    exclude_words: Optional[Set[str]] = None,
    min_len: int = 4,
) -> Set[str]:
    """Extract informative non-stopword tokens from rationale text."""
    exclude = STOP_WORDS if exclude_words is None else STOP_WORDS | exclude_words
    words = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
    return {w for w in words if len(w) >= min_len and w not in exclude}


@dataclass
class AmbossQuestion:
    """Dataclass holding structured Amboss clinical question data."""

    qid: str
    vignette: str
    learning_objective: str
    gold_answer: str
    distractors: List[Dict[str, Any]]
    options: List[Dict[str, Any]]
    gold_why: str = ""

    @property
    def differential_candidates(self) -> List[str]:
        """List of candidate names for the differential (distractor options)."""
        return [d["content"] for d in self.distractors if d.get("content")]

    @property
    def distractor_buts(self) -> Dict[str, str]:
        """Mapping from candidate name to its rule-out rationale."""
        return {
            d["content"]: (d.get("explanationBut") or d.get("rationale") or d.get("explanationWhy") or "")
            for d in self.distractors
            if d.get("content")
        }

    def to_prompt_item(self, include_options: bool = True) -> PromptItem:
        """Convert this Amboss question into a training PromptItem."""
        prompt = self.vignette.strip()
        if include_options and self.options:
            opt_lines = [
                f"{opt['letter'].upper()}. {opt['content']}"
                for opt in self.options
                if opt.get("content")
            ]
            if opt_lines:
                prompt += "\n\nOptions:\n" + "\n".join(opt_lines)

        prompt += (
            "\n\nPlease analyze this clinical case. In your thinking process (<think>...</think>), "
            "evaluate the clinical presentation, develop a differential diagnosis, and rule out competing candidate conditions. "
            "Provide the final diagnosis inside <answer>...</answer>."
        )

        metadata = {
            "qid": self.qid,
            "differential_candidates": self.differential_candidates,
            "learning_objective": self.learning_objective,
            "gold_why": self.gold_why,
            "distractor_buts": self.distractor_buts,
            "options": self.options,
        }

        return PromptItem(
            prompt=prompt,
            ground_truth=self.gold_answer,
            metadata=metadata,
        )


def parse_amboss_json(data: Dict[str, Any], file_stem: str = "") -> AmbossQuestion:
    """Parse raw Amboss JSON dictionary into an AmbossQuestion dataclass."""
    qid = str(data.get("eid") or file_stem)
    vignette = clean_amboss_html(data.get("content", ""))
    learning_objective = clean_amboss_html(data.get("learningObjective", ""))

    gold_answer = ""
    gold_why = ""
    distractors: List[Dict[str, Any]] = []
    options: List[Dict[str, Any]] = []

    for ans in data.get("answers", []):
        content = clean_amboss_html(ans.get("content", ""))
        letter = str(ans.get("letter", "")).strip().lower()
        is_correct = bool(ans.get("isCorrect", False))
        why = clean_amboss_html(ans.get("explanationWhy", ""))
        but = clean_amboss_html(ans.get("explanationBut", ""))
        rationale = but if but else why

        opt_info = {
            "content": content,
            "letter": letter,
            "isCorrect": is_correct,
            "is_correct": is_correct,
            "explanationWhy": why,
            "explanationBut": but,
            "rationale": rationale,
        }
        options.append(opt_info)

        if is_correct:
            gold_answer = content
            gold_why = why if why else but
        else:
            distractors.append(opt_info)

    return AmbossQuestion(
        qid=qid,
        vignette=vignette,
        learning_objective=learning_objective,
        gold_answer=gold_answer,
        distractors=distractors,
        options=options,
        gold_why=gold_why,
    )


def load_amboss_questions(
    directory_path: Union[str, Path],
    max_questions: Optional[int] = None,
    include_options: bool = True,
) -> List[PromptItem]:
    """Loads Amboss questions into PromptItem instances.

    Each PromptItem contains:
    - prompt: Clean clinical vignette formatted to elicit a <think> differential and <answer> diagnosis.
    - ground_truth: Gold answer text.
    - metadata: Contains differential_candidates, learning_objective, gold_why, distractor_buts.

    Args:
        directory_path: Directory containing Amboss JSON files.
        max_questions: Optional maximum number of questions to load.
        include_options: Whether to include multiple-choice options in the formatted prompt.

    Returns:
        List of PromptItem instances ready for DiffuGRPO RL training.
    """
    path = Path(directory_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Amboss questions directory not found: {path}")

    json_files = sorted(path.glob("*.json"))
    items: List[PromptItem] = []

    for fpath in json_files:
        if max_questions is not None and len(items) >= max_questions:
            break
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            q = parse_amboss_json(data, file_stem=fpath.stem)
            items.append(q.to_prompt_item(include_options=include_options))
        except Exception:
            # Skip unparseable files gracefully
            continue

    return items


class AmbossDifferentialReward(BaseReward):
    """Hardened clinical differential scoring reward.

    Scoring logic & exploit protections:
    1. Candidate Differential & Gold Correctness (+1.0):
       - Recognizes gold diagnosis in <answer> tags OR within a <differential> candidate block.
       - Supports option letter matching when options are present.
    2. Gated Rule-Out Credit (+0.25 per verified distractor, max +1.0):
       - Gating: If gold is not in the differential or answer, rule-out credit is strictly gated
         (prevents "articulate-but-wrong" keyword harvesting).
       - Structured/Window Attribution: Rationale keywords must appear within proximity
         (+/- 25 words) of the specific candidate distractor, or within a <rule_out target="..."> block.
       - Negation Scope Check: If distractor rationale relies on absent findings (e.g. "no lymphadenopathy"),
         affirmative statements ("prominent lymphadenopathy", "severe lymphadenopathy") are invalidated.
    3. Discrete Bounding:
       - 1.0 for gold + 0.25 * n_verified_distractors (up to 4) = max 2.0.
       - The 2.0 ceiling is only reachable by a complete, correct differential.
    """

    def __init__(
        self,
        gold_reward: float = 1.0,
        ruleout_credit_per_candidate: float = 0.25,
        max_ruleout_credit: float = 1.0,
        min_keyword_length: int = 4,
        gate_on_gold: bool = True,
        proximity_word_window: int = 25,
    ):
        self.gold_reward = gold_reward
        self.ruleout_credit_per_candidate = ruleout_credit_per_candidate
        self.max_ruleout_credit = max_ruleout_credit
        self.min_keyword_length = min_keyword_length
        self.gate_on_gold = gate_on_gold
        self.proximity_word_window = proximity_word_window

        self.xml_answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        self.xml_diff_pattern = re.compile(r"<differential>(.*?)</differential>", re.DOTALL | re.IGNORECASE)
        self.xml_ruleout_pattern = re.compile(r"<rule_out\s+target=[\"'](.*?)[\"']>(.*?)</rule_out>", re.DOTALL | re.IGNORECASE)

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer string from <answer> tags if present."""
        matches = self.xml_answer_pattern.findall(text)
        if matches:
            return matches[-1].strip()
        return None

    def extract_differential(self, text: str) -> Optional[str]:
        """Extract differential content from <differential> tags if present."""
        matches = self.xml_diff_pattern.findall(text)
        if matches:
            return " ".join(matches).strip()
        return None

    def _match_gold(
        self,
        completion: str,
        ground_truth: str,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Determine whether the gold diagnosis matches the answer or differential."""
        norm_gt = _normalize_text(ground_truth)
        if not norm_gt:
            return False

        # Check gold option letter if available
        gold_letter = ""
        if options:
            for opt in options:
                if opt.get("isCorrect") or opt.get("is_correct"):
                    gold_letter = str(opt.get("letter", "")).strip().lower()
                    break

        # 1. Check <answer> tag
        extracted = self.extract_answer(completion)
        if extracted is not None:
            norm_ext = _normalize_text(extracted)
            if norm_ext:
                if norm_gt in norm_ext or (len(norm_ext) >= 4 and norm_ext in norm_gt):
                    return True
                if gold_letter:
                    tokens = set(norm_ext.split())
                    if gold_letter in tokens or norm_ext == gold_letter:
                        return True

        # 2. Check <differential> tag (candidate set recall)
        diff_text = self.extract_differential(completion)
        if diff_text is not None:
            norm_diff = _normalize_text(diff_text)
            if norm_gt in norm_diff:
                return True
            if gold_letter:
                diff_tokens = set(norm_diff.split())
                if gold_letter in diff_tokens:
                    return True

        # 3. Fallback: check completion text directly
        norm_comp = _normalize_text(completion)
        return norm_gt in norm_comp

    def _is_negation_violated(self, rationale: str, context_text: str, keyword: str) -> bool:
        """Check if rationale specifies an absence but context affirms presence."""
        rat_lower = rationale.lower()
        neg_indicators = ["no ", "without ", "lacks ", "absent ", "absence of ", "negative "]
        is_negated_rationale = any(ind in rat_lower for ind in neg_indicators)
        if not is_negated_rationale:
            return False

        # If rationale states absence, check if context used affirmative modifiers directly before keyword
        ctx_lower = context_text.lower()
        kw_idx = ctx_lower.find(keyword)
        if kw_idx > 0:
            prefix = ctx_lower[max(0, kw_idx - 35) : kw_idx]
            affirm_words = ["prominent", "marked", "severe", "present", "presence of", "shows", "confirmed", "diffuse"]
            if any(aw in prefix for aw in affirm_words):
                return True
        return False

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        token_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> float:
        """Compute hardened scalar clinical differential reward."""
        ground_truth = kwargs.get("ground_truth") or kwargs.get("target") or kwargs.get("gold_answer")
        differential_candidates = kwargs.get("differential_candidates") or []
        distractor_buts = kwargs.get("distractor_buts") or {}
        options = kwargs.get("options") or None

        # 1. Gold diagnosis match (+1.0)
        gold_matched = False
        if ground_truth:
            gold_matched = self._match_gold(completion, str(ground_truth), options=options)

        # 2. Gating: If gold is not matched and gating is enabled, reject rule-out credit
        if self.gate_on_gold and not gold_matched:
            return 0.0

        # 3. Parse candidate rule-outs
        items: List[Tuple[str, Any]] = []
        if isinstance(distractor_buts, dict):
            items = list(distractor_buts.items())
        elif isinstance(distractor_buts, list):
            if differential_candidates and len(differential_candidates) == len(distractor_buts):
                items = list(zip(differential_candidates, distractor_buts))
            else:
                items = [("", r) for r in distractor_buts]
        elif isinstance(distractor_buts, str):
            items = [("", distractor_buts)]

        # Extract reasoning text (preferring <think> tags, or full completion)
        xml_think_matches = re.findall(r"<think>(.*?)</think>", completion, re.DOTALL | re.IGNORECASE)
        if xml_think_matches:
            reasoning_text = " ".join(xml_think_matches)
        else:
            reasoning_text = re.sub(r"(?is)<answer>.*?</answer>", " ", completion)

        reasoning_words = re.findall(r"[a-zA-Z0-9_\-]+", reasoning_text.lower())
        reasoning_lower = reasoning_text.lower()

        # Parse any explicit <rule_out target="..."> blocks
        explicit_ruleouts = {}
        for target, content in self.xml_ruleout_pattern.findall(completion):
            explicit_ruleouts[_normalize_text(target)] = content.lower()

        extracted_answer = self.extract_answer(completion)
        extracted_norm = _normalize_text(extracted_answer) if extracted_answer else ""

        matched_ruleouts = 0

        for cand, rationale in items:
            cand_str = str(cand).strip()
            cand_norm = _normalize_text(cand_str)
            cand_words = set(re.findall(r"[a-zA-Z0-9_\-]+", cand_str.lower()))

            # If candidate was explicitly chosen as the final answer, it cannot be ruled out
            if extracted_norm and cand_norm and (cand_norm in extracted_norm or extracted_norm in cand_norm):
                continue

            # Extract discriminative keywords from the rationale
            if isinstance(rationale, (list, set, tuple)):
                kws = {
                    str(k).lower().strip()
                    for k in rationale
                    if len(str(k).strip()) >= self.min_keyword_length
                }
            else:
                kws = _extract_keywords(
                    str(rationale),
                    exclude_words=cand_words,
                    min_len=self.min_keyword_length,
                )

            if not kws:
                continue

            # A. Check explicit <rule_out target="..."> block if present
            distractor_verified = False
            for target_norm, block_content in explicit_ruleouts.items():
                if cand_norm in target_norm or target_norm in cand_norm:
                    matching_in_block = [kw for kw in kws if kw in block_content]
                    if matching_in_block:
                        if not any(self._is_negation_violated(str(rationale), block_content, kw) for kw in matching_in_block):
                            distractor_verified = True
                            break

            # B. Check proximity window around candidate mention in reasoning_text
            if not distractor_verified and cand_str:
                # Find all word positions where candidate is mentioned
                cand_token = list(cand_words - STOP_WORDS)[0] if (cand_words - STOP_WORDS) else ""
                if cand_token:
                    mention_indices = [i for i, w in enumerate(reasoning_words) if w == cand_token]
                    for idx in mention_indices:
                        start_pos = max(0, idx - self.proximity_word_window)
                        end_pos = min(len(reasoning_words), idx + self.proximity_word_window)
                        window_text = " ".join(reasoning_words[start_pos:end_pos])
                        matching_kws = [kw for kw in kws if kw in window_text]
                        if matching_kws:
                            if not any(self._is_negation_violated(str(rationale), window_text, kw) for kw in matching_kws):
                                distractor_verified = True
                                break

            # C. Fallback: if candidate name is not in prompt metadata, require at least 2 keywords
            if not distractor_verified and not cand_str:
                matching_kws = [kw for kw in kws if kw in reasoning_lower]
                if len(matching_kws) >= 2:
                    if not any(self._is_negation_violated(str(rationale), reasoning_lower, kw) for kw in matching_kws):
                        distractor_verified = True

            if distractor_verified:
                matched_ruleouts += 1

        ruleout_score = min(
            self.max_ruleout_credit,
            matched_ruleouts * self.ruleout_credit_per_candidate,
        )

        total_reward = (self.gold_reward if gold_matched else 0.0) + ruleout_score
        return max(0.0, min(2.0, float(total_reward)))
