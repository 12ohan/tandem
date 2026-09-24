from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

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

# Grammatical-only stop words for diagnosis matching (preserves clinical head nouns like syndrome, disease)
DIAGNOSIS_STOP_WORDS: Set[str] = {
    "a", "an", "the", "of", "in", "and", "or", "with", "due", "to", "by", "for", "on", "at", "from",
}

# Short clinical tokens to preserve during keyword extraction (raw-case checked)
SHORT_CLINICAL_ALLOWLIST: Set[str] = {
    "PH", "PO2", "PCO2", "BP", "O2", "HB",
}

# Generic clinical head nouns that cannot qualify as a diagnosis on their own
GENERIC_CLINICAL_HEAD_NOUNS: Set[str] = {
    "syndrome", "syndromes", "disease", "diseases", "disorder", "disorders",
    "condition", "conditions", "infection", "infections", "carcinoma",
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


def _diagnosis_tokens(text: str) -> Set[str]:
    """Tokenise a diagnosis, preserving clinical head nouns and expanding common abbreviations."""
    s = text.lower()
    # Possessive normalization: Cushing's -> cushing, Cushing’s -> cushing
    s = re.sub(r"['’]s\b", "", s)
    # Minimal abbreviation expansion for the current positive-control contract.
    s = re.sub(r"\bs\.\s*", "streptococcus ", s)
    s = re.sub(r"\bstrep\.?\s*", "streptococcus ", s)
    tokens = set(re.findall(r"[a-zA-Z0-9]+", s)) - DIAGNOSIS_STOP_WORDS
    return tokens


def _extract_keywords(
    text: str,
    exclude_words: Optional[Set[str]] = None,
    min_len: int = 4,
) -> Set[str]:
    """Extract informative non-stopword tokens from rationale text."""
    exclude = STOP_WORDS if exclude_words is None else STOP_WORDS | exclude_words
    raw_words = re.findall(r"[a-zA-Z0-9_\-]+", text)
    kept: Set[str] = set()
    for raw in raw_words:
        lowered = raw.lower()
        allowed = len(raw) >= min_len or raw.upper() in SHORT_CLINICAL_ALLOWLIST
        if allowed and lowered not in exclude:
            kept.add(lowered)
    return kept


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
    def distractor_options(self) -> List[str]:
        """Explicit alias for differential_candidates (distractors only)."""
        return self.differential_candidates

    @property
    def all_candidates(self) -> List[str]:
        """Complete candidate set including gold diagnosis and all distractor options."""
        cands = []
        if self.gold_answer:
            cands.append(self.gold_answer)
        for d in self.distractors:
            name = d.get("content")
            if name and name not in cands:
                cands.append(name)
        return cands

    @property
    def distractor_buts(self) -> Dict[str, str]:
        """Mapping from candidate name to its rule-out rationale."""
        return {
            d["content"]: (d.get("explanationBut") or d.get("rationale") or d.get("explanationWhy") or "")
            for d in self.distractors
            if d.get("content")
        }

    def to_prompt_item(
        self, include_options: bool = True, format_primed: bool = False
    ) -> PromptItem:
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

        if format_primed:
            prompt += (
                "\n\nPlease analyze this clinical case. Structure your response using <think>...</think> "
                "for clinical reasoning and distractor rule-outs, followed by <answer>...</answer> for your final diagnosis.\n"
                "Example format:\n"
                "<think>\n"
                "Evaluating presentation: patient presents with acute symptom complex.\n"
                "Differential diagnosis includes Condition A and Condition B.\n"
                "Condition A is ruled out because [pertinent negative finding is absent].\n"
                "Condition B is the primary diagnosis supported by [pertinent positive finding].\n"
                "</think>\n"
                "<answer>Condition B</answer>\n\n"
                "Now analyze the patient case above:"
            )
        else:
            prompt += (
                "\n\nPlease analyze this clinical case. In your thinking process (<think>...</think>), "
                "evaluate the clinical presentation, develop a differential diagnosis, and rule out competing candidate conditions. "
                "Provide the final diagnosis inside <answer>...</answer>."
            )

        metadata = {
            "qid": self.qid,
            "differential_candidates": self.differential_candidates,
            "all_candidates": self.all_candidates,
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
    split: Literal["all", "train", "eval"] = "all",
    eval_split_ratio: float = 0.15,
    eval_seed: int = 42,
    format_primed: bool = False,
) -> List[PromptItem]:
    """Loads Amboss questions into PromptItem instances with deterministic train/eval splitting.

    Each PromptItem contains:
    - prompt: Clean clinical vignette formatted to elicit a <think> differential and <answer> diagnosis.
    - ground_truth: Gold answer text.
    - metadata: Contains differential_candidates, all_candidates, learning_objective, gold_why, distractor_buts.

    Args:
        directory_path: Directory containing Amboss JSON files.
        max_questions: Optional maximum number of questions to load.
        include_options: Whether to include multiple-choice options in the formatted prompt.
        split: "all", "train" (85% default), or "eval" (15% held-out default).
        eval_split_ratio: Fraction of questions allocated to held-out eval (default 0.15 ~ 500 Qs).
        eval_seed: Seed for deterministic hash-based splitting on question ID.
        format_primed: Whether to include in-context clinical format priming exemplar.

    Returns:
        List of PromptItem instances ready for DiffuGRPO RL training or held-out evaluation.
    """
    import hashlib

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

            # Deterministic hash split on QID
            if split != "all":
                hash_val = int(hashlib.sha256(f"{q.qid}_{eval_seed}".encode()).hexdigest()[:8], 16)
                is_eval = (hash_val % 10000) < int(eval_split_ratio * 10000)
                if split == "eval" and not is_eval:
                    continue
                if split == "train" and is_eval:
                    continue

            items.append(
                q.to_prompt_item(include_options=include_options, format_primed=format_primed)
            )
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
       - 2.0 for gold + 0.20 * n_verified_distractors (up to 4) = max 2.80.
       - The 2.80 ceiling is only reachable by a complete, correct differential.
       - Under magnitude dominance (gate_on_gold=False default), any wrong answer
         scores at most 0.80, strictly dominated by any correct answer (2.00).
    """

    def __init__(
        self,
        gold_reward: float = 2.0,
        ruleout_credit_per_candidate: float = 0.20,
        max_ruleout_credit: float = 0.80,
        min_keyword_length: int = 3,
        gate_on_gold: bool = False,
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
        """Determine whether the gold diagnosis matches the answer or differential.

        Strict clinical matching rules:
        1. Primary check: gold option letter (e.g. 'b' or '(b)' or 'option b') in <answer>.
        2. Exact match: normalized ground truth == normalized extracted text.
        3. Word-bounded full phrase match: ground truth phrase appears with complete word boundaries
           in <answer> or <differential> (e.g. r'\b' + re.escape(gt) + r'\b').
           Prevents 'pain' matching 'painless'.
        4. Token-level partial match (with length ratio >= 0.50):
           All content tokens of the extracted candidate must be a non-empty subset of the gold diagnosis tokens.
           Prevents 'cystitis' matching 'cholecystitis', 'carditis' matching 'myocarditis' or 'pericarditis',
           while allowing legitimate compound partials like 'Cholecystitis' for 'Acute cholecystitis'.
        5. STRICT: NO fallback to general completion/<think> text.
        """
        gt_clean = ground_truth.strip()
        if not gt_clean:
            return False

        gt_lower = gt_clean.lower()
        norm_gt = _normalize_text(gt_clean)
        gt_tokens = _diagnosis_tokens(gt_clean)
        if not gt_tokens:
            gt_tokens = set(re.findall(r"[a-zA-Z0-9]+", gt_lower))

        # Extract gold option letter if available
        gold_letter = ""
        if options:
            for opt in options:
                if opt.get("isCorrect") or opt.get("is_correct"):
                    gold_letter = str(opt.get("letter", "")).strip().lower()
                    break

        def _is_text_matching_gold(candidate_text: str) -> bool:
            cand_strip = candidate_text.strip()
            if not cand_strip:
                return False
            cand_lower = cand_strip.lower()
            norm_cand = _normalize_text(cand_strip)

            # A. Option letter match in candidate text
            if gold_letter:
                tokens = set(re.findall(r"[a-zA-Z0-9]+", cand_lower))
                if gold_letter in tokens or norm_cand == gold_letter:
                    return True
                if f"({gold_letter})" in cand_lower or f"option {gold_letter}" in cand_lower:
                    return True

            # B. Exact string match (ignoring punctuation/case)
            if norm_gt and norm_cand and norm_gt == norm_cand:
                return True

            # C. Word-bounded full phrase match
            # e.g. "Streptococcus pneumoniae" in "The diagnosis is Streptococcus pneumoniae"
            # Crucially prevents "pain" matching "painless" via \b
            if re.search(rf"\b{re.escape(gt_lower)}\b", cand_lower):
                return True

            # D. Token-level subset partial match for multi-word diseases
            cand_tokens = _diagnosis_tokens(cand_strip)
            if not cand_tokens:
                cand_tokens = set(re.findall(r"[a-zA-Z0-9]+", cand_lower))

            # Reject if candidate tokens consist solely of generic clinical head nouns
            if cand_tokens.issubset(GENERIC_CLINICAL_HEAD_NOUNS):
                return False

            if cand_tokens and cand_tokens.issubset(gt_tokens):
                char_ratio = sum(len(t) for t in cand_tokens) / max(1, sum(len(t) for t in gt_tokens))
                if char_ratio >= 0.50:
                    return True

            return False

        # 1. Check <answer> tag
        extracted_answer = self.extract_answer(completion)
        if extracted_answer is not None and _is_text_matching_gold(extracted_answer):
            return True

        # 2. Check <differential> tag
        diff_text = self.extract_differential(completion)
        if diff_text is not None and _is_text_matching_gold(diff_text):
            return True

        return False

    def _is_negation_violated(self, rationale: str, context_text: str, keyword: str) -> bool:
        """Check if rationale specifies an absence but context affirms presence."""
        rat_lower = rationale.lower()
        neg_indicators = ["no ", "without ", "lacks ", "absent ", "absence of ", "negative ", "denies "]
        is_negated_rationale = any(ind in rat_lower for ind in neg_indicators)
        if not is_negated_rationale:
            return False

        # If rationale states absence, check if context used affirmative modifiers directly before keyword
        ctx_lower = context_text.lower()
        affirm_patterns = [
            r"\bprominent\b", r"\bmarked\b", r"\bsevere\b", r"\bpresent\b", r"\bpresence\b",
            r"\bshows\b", r"\bconfirmed\b", r"\bdiffuse\b", r"\breveals\b", r"\bdemonstrates\b",
            r"\bnotable\b",
        ]
        # Iterate over all occurrences of keyword in context, checking each preceding window
        kw_indices = [m.start() for m in re.finditer(rf"\b{re.escape(keyword.lower())}\b", ctx_lower)]
        for kw_idx in kw_indices:
            prefix = ctx_lower[max(0, kw_idx - 40) : kw_idx]
            if any(re.search(pat, prefix) for pat in affirm_patterns):
                return True
        return False

    def _has_nearby_contrast(self, model_text: str, matching_kws: List[str], window: int) -> bool:
        """Return True if a contrast marker is within `window` words of a matched keyword."""
        words = re.findall(r"[a-zA-Z0-9_\-]+", model_text.lower())
        if not words or not matching_kws:
            return False
        contrast_words = {
            "but", "however", "unlike", "whereas", "not", "no", "without",
            "lacks", "absence", "absent", "denies", "unlikely", "less", "lower",
        }
        kw_set = {str(k).lower() for k in matching_kws}
        contrast_idx = [i for i, w in enumerate(words) if w in contrast_words]
        if not contrast_idx:
            return False
        kw_idx = [i for i, w in enumerate(words) if w in kw_set]
        return any(abs(k - c) <= window for k in kw_idx for c in contrast_idx)

    def _is_ruleout_valid(self, rationale: str, model_text: str, matching_kws: List[str], require_proximity: bool = False) -> bool:
        """Validate that the model's reasoning establishes legitimate distractor rule-out.

        1. Contrast/Negation in Model Text:
           The model cannot merely regurgitate distractor features (e.g. 'H. ducreyi: painful ulcers').
           The model's text must contain a contrast or negation marker indicating that the patient
           lacks the distractor's features or that the distractor is contrasted against the patient:
           {but, however, unlike, whereas, not, no, without, lacks, absent, absence, denies, unlikely}.
           Crucially, this is checked on the MODEL's text (not the AMBOSS rationale). Even if the
           AMBOSS rationale contains 'unlike' or 'no', the model itself must provide the contrast/negation.

        2. Negation Scope Check (Absence Violation):
           If the distractor rationale explicitly states an absence (e.g. 'no lymphadenopathy'),
           the model must NOT affirm presence ('shows prominent lymphadenopathy').
        """
        # 1. Check if the distractor rationale asserts an absence that the model affirmed
        for kw in matching_kws:
            if self._is_negation_violated(rationale, model_text, kw):
                return False

        # 2. The model's own text must contain contrast or negation markers
        contrast_patterns = [
            r"\bbut\b", r"\bhowever\b", r"\bunlike\b", r"\bwhereas\b",
            r"\bnot\b", r"\bno\b", r"\bwithout\b", r"\blacks\b",
            r"\babsence of\b", r"\babsence\b", r"\babsent\b", r"\bdenies\b",
            r"\bunlikely\b", r"\bless\b", r"\blower\b",
        ]
        if require_proximity:
            return self._has_nearby_contrast(model_text, matching_kws, self.proximity_word_window)
        model_lower = model_text.lower()
        return any(re.search(pat, model_lower) for pat in contrast_patterns)

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
                    str(k).strip().lower()
                    for k in rationale
                    if len(str(k).strip()) >= self.min_keyword_length
                    or str(k).strip().upper() in SHORT_CLINICAL_ALLOWLIST
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
                    matching_in_block = [
                        kw for kw in kws
                        if re.search(rf"\b{re.escape(kw)}\b", block_content)
                    ]
                    if matching_in_block and self._is_ruleout_valid(str(rationale), block_content, matching_in_block):
                        distractor_verified = True
                        break

            # B. Check proximity window around candidate mention in reasoning_text
            if not distractor_verified and cand_str:
                cand_tokens = [w for w in cand_words if w not in STOP_WORDS and len(w) >= 3]
                if cand_tokens:
                    mention_indices = [i for i, w in enumerate(reasoning_words) if w in cand_tokens]
                    for idx in mention_indices:
                        start_pos = max(0, idx - self.proximity_word_window)
                        end_pos = min(len(reasoning_words), idx + self.proximity_word_window)
                        window_text = " ".join(reasoning_words[start_pos:end_pos])
                        matching_kws = [
                            kw for kw in kws
                            if re.search(rf"\b{re.escape(kw)}\b", window_text)
                        ]
                        if matching_kws and self._is_ruleout_valid(str(rationale), window_text, matching_kws, require_proximity=True):
                            distractor_verified = True
                            break

            # C. Fallback: if candidate name is not in prompt metadata, require at least 2 keywords
            if not distractor_verified and not cand_str:
                matching_kws = [
                    kw for kw in kws
                    if re.search(rf"\b{re.escape(kw)}\b", reasoning_lower)
                ]
                if len(matching_kws) >= 2 and self._is_ruleout_valid(str(rationale), reasoning_lower, matching_kws, require_proximity=True):
                    distractor_verified = True

            if distractor_verified:
                matched_ruleouts += 1

        ruleout_score = min(
            self.max_ruleout_credit,
            matched_ruleouts * self.ruleout_credit_per_candidate,
        )

        total_reward = (self.gold_reward if gold_matched else 0.0) + ruleout_score
        max_cap = self.gold_reward + self.max_ruleout_credit
        return max(0.0, min(max_cap, float(total_reward)))
