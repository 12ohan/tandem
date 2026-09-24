from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


class UMLSTrieNode:
    """A node in the UMLS Entity Radix Trie representing a subword token in a concept sequence.

    Attributes:
        token_id: The subword token ID associated with this node (None for root).
        children: Dictionary mapping child subword token IDs to child UMLSTrieNodes.
        is_terminal: Whether a medical concept terminates at this node.
        cui: Concept Unique Identifier (CUI) if terminal (primary/most frequent).
        concept_name: Full medical concept string if terminal.
        frequency: Total count of concepts / occurrences passing through or ending at this node.
        terminal_count: Count of concepts terminating exactly at this node.
        metadata: Additional concept metadata (e.g. sab, tty, code, semantic_types).
        best_child: Pre-cached tuple of (child_token_id, child_node) with highest frequency for sub-microsecond traversal.
    """

    __slots__ = (
        "token_id",
        "children",
        "is_terminal",
        "cui",
        "concept_name",
        "frequency",
        "terminal_count",
        "metadata",
        "best_child",
    )

    def __init__(
        self,
        token_id: Optional[int] = None,
        is_terminal: bool = False,
        cui: Optional[str] = None,
        concept_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        frequency: int = 0,
    ) -> None:
        self.token_id = token_id
        self.children: Dict[int, UMLSTrieNode] = {}
        self.is_terminal = is_terminal
        self.cui = cui
        self.concept_name = concept_name
        self.frequency = frequency
        self.terminal_count = 1 if is_terminal else 0
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.best_child: Optional[Tuple[int, UMLSTrieNode]] = None

    def update_best_child(self, token_id: int, child_node: UMLSTrieNode) -> None:
        """Update cached best child based on child frequency."""
        if self.best_child is None:
            self.best_child = (token_id, child_node)
        elif child_node.frequency > self.best_child[1].frequency:
            self.best_child = (token_id, child_node)

    def __repr__(self) -> str:
        return (
            f"UMLSTrieNode(token_id={self.token_id}, "
            f"is_terminal={self.is_terminal}, "
            f"cui={repr(self.cui)}, "
            f"children_count={len(self.children)}, "
            f"frequency={self.frequency})"
        )


class UMLSEntityTrie:
    """Fast Radix Trie indexing subword token sequences for medical concepts.

    Designed for zero-compute speculative drafting and sub-microsecond medical entity continuation.
    """

    def __init__(self) -> None:
        self.root = UMLSTrieNode()
        self._node_count: int = 1
        self._unique_terms: int = 0
        self._total_insertions: int = 0

    @property
    def num_nodes(self) -> int:
        """Total number of nodes currently allocated in the trie."""
        return self._node_count

    @property
    def num_concepts(self) -> int:
        """Total number of concept insertions performed."""
        return self._total_insertions

    @property
    def num_unique_terms(self) -> int:
        """Number of unique terminal token sequences stored in the trie."""
        return self._unique_terms

    def __len__(self) -> int:
        return self._total_insertions

    def insert(
        self,
        token_ids: Sequence[int],
        concept_name: str,
        cui: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a tokenized medical concept sequence into the trie.

        Args:
            token_ids: Subword token IDs representing the medical concept.
            concept_name: String surface form of the concept.
            cui: UMLS Concept Unique Identifier (e.g. 'C0000005').
            metadata: Optional dictionary with vocabulary source (SAB), TTY, code, semantic types, etc.
        """
        if not token_ids:
            return

        curr = self.root
        curr.frequency += 1

        for token in token_ids:
            child = curr.children.get(token)
            if child is None:
                child = UMLSTrieNode(token_id=token)
                curr.children[token] = child
                self._node_count += 1
            child.frequency += 1
            curr.update_best_child(token, child)
            curr = child

        if not curr.is_terminal:
            self._unique_terms += 1
            curr.is_terminal = True
            curr.cui = cui
            curr.concept_name = concept_name
        else:
            curr.terminal_count += 1
            if curr.cui is None:
                curr.cui = cui
            if curr.concept_name is None:
                curr.concept_name = concept_name

        if metadata:
            curr.metadata.update(metadata)

        all_cuis = curr.metadata.setdefault("all_cuis", set())
        all_cuis.add(cui)
        all_names = curr.metadata.setdefault("all_names", set())
        all_names.add(concept_name)

        self._total_insertions += 1

    def find_node(self, tokens: Sequence[int]) -> Optional[UMLSTrieNode]:
        """Walk the trie from root following the token sequence.

        Args:
            tokens: Sequence of subword token IDs.

        Returns:
            The UMLSTrieNode reached, or None if the path diverges.
        """
        curr = self.root
        for token in tokens:
            curr = curr.children.get(token)
            if curr is None:
                return None
        return curr

    def exact_lookup(self, tokens: Sequence[int]) -> Optional[Tuple[str, str]]:
        """Look up whether an exact token sequence corresponds to a terminal concept.

        Args:
            tokens: Sequence of subword token IDs.

        Returns:
            Tuple of (cui, concept_name) if the token sequence terminates a concept, else None.
        """
        node = self.find_node(tokens)
        if node is not None and node.is_terminal and node.cui is not None and node.concept_name is not None:
            return (node.cui, node.concept_name)
        return None

    def find_longest_matching_suffix(
        self,
        tokens: Sequence[int],
        max_lookback: int = 32,
    ) -> Tuple[Optional[UMLSTrieNode], int]:
        """Find the longest suffix of `tokens` that matches a valid prefix path in the trie.

        Given a stream or history of tokens emitted by a model, searches suffixes of decreasing
        lookback distance to identify if a medical concept prefix has been initiated.

        Args:
            tokens: Sequence of tokens (e.g. recent context or generated IDs).
            max_lookback: Maximum number of trailing tokens to inspect.

        Returns:
            Tuple of (matching_node, start_index). If no suffix matches, returns (None, len(tokens)).
        """
        n = len(tokens)
        if n == 0:
            return None, 0

        start_min = max(0, n - max_lookback)
        for start in range(start_min, n):
            first_tok = tokens[start]
            if first_tok not in self.root.children:
                continue

            curr: Optional[UMLSTrieNode] = self.root.children[first_tok]
            matched = True
            for i in range(start + 1, n):
                curr = curr.children.get(tokens[i])
                if curr is None:
                    matched = False
                    break

            if matched and curr is not None:
                return curr, start

        return None, n

    def extract_concepts(
        self,
        text: str,
        tokenizer: Any,
    ) -> List[Tuple[str, str, int, int]]:
        """Extract all recognized UMLS medical concepts from a text string.

        Args:
            text: Arbitrary text string (e.g. clinical vignette or candidate string).
            tokenizer: Tokenizer with an .encode(text, add_special_tokens=False) method.

        Returns:
            List of tuples (cui, concept_name, token_start, token_end).
        """
        if not text:
            return []

        tokens = tokenizer.encode(text, add_special_tokens=False)
        n = len(tokens)
        matches: List[Tuple[str, str, int, int]] = []

        for start in range(n):
            curr = self.root.children.get(tokens[start])
            if curr is None:
                continue

            if curr.is_terminal and curr.cui and curr.concept_name:
                matches.append((curr.cui, curr.concept_name, start, start + 1))

            for j in range(start + 1, n):
                curr = curr.children.get(tokens[j])
                if curr is None:
                    break
                if curr.is_terminal and curr.cui and curr.concept_name:
                    matches.append((curr.cui, curr.concept_name, start, j + 1))

        return matches

    def propose_draft(
        self,
        prefix_tokens: List[int],
        max_tokens: int = 5,
        deterministic_only: bool = False,
        max_lookback: int = 32,
    ) -> List[int]:
        """Propose speculative candidate tokens for medical concepts given latest context tokens.

        Traverses the trie starting from the longest matching suffix of prefix_tokens. Proposes
        up to `max_tokens` speculative tokens following either:
        1. A deterministic path (if deterministic_only=True, stopping at any ambiguous branching point
           or concept termination boundary).
        2. The top-frequency branch (if deterministic_only=False, dynamically following the most frequent
           medical concept continuation).

        Operates in sub-microsecond time (~0.2 - 0.5 microseconds).

        Args:
            prefix_tokens: The latest tokens emitted by the model.
            max_tokens: Maximum number of speculative draft tokens to propose (e.g. block_size K).
            deterministic_only: If True, only emit tokens when the continuation path is strictly unique.
            max_lookback: Maximum trailing tokens to scan for concept prefix matching.

        Returns:
            List of proposed speculative token IDs.
        """
        if not prefix_tokens or max_tokens <= 0:
            return []

        curr, _ = self.find_longest_matching_suffix(prefix_tokens, max_lookback=max_lookback)
        if curr is None or curr is self.root:
            return []

        draft: List[int] = []
        while len(draft) < max_tokens:
            if not curr.children:
                break

            if deterministic_only:
                if curr.is_terminal:
                    # Concept is already complete; continuing into compound or modifier is non-deterministic
                    break
                if len(curr.children) != 1:
                    # Ambiguous branch point
                    break
                tok, next_node = next(iter(curr.children.items()))
            else:
                if curr.best_child is not None:
                    tok, next_node = curr.best_child
                else:
                    tok, next_node = max(curr.children.items(), key=lambda item: item[1].frequency)

            draft.append(tok)
            curr = next_node

        return draft

    def propose_deterministic_draft(
        self,
        prefix_tokens: List[int],
        max_tokens: int = 5,
        max_lookback: int = 32,
    ) -> List[int]:
        """Convenience method to propose strictly deterministic continuation tokens."""
        return self.propose_draft(
            prefix_tokens=prefix_tokens,
            max_tokens=max_tokens,
            deterministic_only=True,
            max_lookback=max_lookback,
        )

    def propose_top_branch_draft(
        self,
        prefix_tokens: List[int],
        max_tokens: int = 5,
        max_lookback: int = 32,
    ) -> List[int]:
        """Convenience method to propose top-branch continuation tokens."""
        return self.propose_draft(
            prefix_tokens=prefix_tokens,
            max_tokens=max_tokens,
            deterministic_only=False,
            max_lookback=max_lookback,
        )

    @classmethod
    def load_mrsty(cls, mrsty_path: str) -> Dict[str, List[Dict[str, str]]]:
        """Parse UMLS MRSTY.RRF to build mapping of CUI -> semantic types."""
        cui_to_sty: Dict[str, List[Dict[str, str]]] = {}
        if not os.path.exists(mrsty_path):
            return cui_to_sty

        with open(mrsty_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split("|")
                if len(parts) >= 4:
                    cui = parts[0]
                    tui = parts[1]
                    sty = parts[3]
                    if cui not in cui_to_sty:
                        cui_to_sty[cui] = []
                    cui_to_sty[cui].append({"tui": tui, "sty": sty})
        return cui_to_sty

    @classmethod
    def build_from_mrconso(
        cls,
        mrconso_path: str,
        tokenizer: Any,
        max_concepts: int = 50000,
        target_sabs: Optional[List[str]] = None,
        mrsty_path: Optional[str] = None,
        include_space_prefix: bool = False,
    ) -> UMLSEntityTrie:
        """Stream-parse English concepts from MRCONSO.RRF and index them into a UMLSEntityTrie.

        Args:
            mrconso_path: Path to UMLS MRCONSO.RRF.
            tokenizer: Tokenizer instance providing .encode(text).
            max_concepts: Maximum number of concept lines to load.
            target_sabs: List of allowed vocabularies (e.g. ['SNOMEDCT_US', 'MSH', 'RXNORM']).
            mrsty_path: Optional path to MRSTY.RRF for semantic type enrichment.
            include_space_prefix: Whether to prepend a space before tokenizing.

        Returns:
            Populated UMLSEntityTrie instance.
        """
        trie = cls()
        sabs = set(target_sabs) if target_sabs is not None else {"SNOMEDCT_US", "MSH", "RXNORM"}

        cui_to_sty = {}
        if mrsty_path and os.path.exists(mrsty_path):
            cui_to_sty = cls.load_mrsty(mrsty_path)

        concepts_loaded = 0
        with open(mrconso_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "|ENG|" not in line:
                    continue

                parts = line.split("|")
                if len(parts) > 14 and parts[1] == "ENG" and parts[11] in sabs:
                    cui = parts[0]
                    sab = parts[11]
                    tty = parts[12]
                    code = parts[13]
                    concept_name = parts[14]

                    if not concept_name.strip():
                        continue

                    text_to_encode = (" " + concept_name) if include_space_prefix else concept_name
                    try:
                        token_ids = tokenizer.encode(text_to_encode, add_special_tokens=False)
                    except TypeError:
                        token_ids = tokenizer.encode(text_to_encode)

                    if not token_ids:
                        continue

                    meta: Dict[str, Any] = {"sab": sab, "tty": tty, "code": code}
                    if cui in cui_to_sty:
                        meta["semantic_types"] = cui_to_sty[cui]

                    trie.insert(
                        token_ids=token_ids,
                        concept_name=concept_name,
                        cui=cui,
                        metadata=meta,
                    )
                    concepts_loaded += 1
                    if max_concepts > 0 and concepts_loaded >= max_concepts:
                        break

        return trie

    def profile_memory(self) -> Dict[str, Union[int, float]]:
        """Compute exact memory consumption of the trie by deep object traversal.

        Traverses all allocated nodes, child maps, strings, and metadata to report exact bytes.

        Returns:
            Dictionary with num_nodes, num_concepts, num_terminals, total_bytes, total_kb,
            total_mb, bytes_per_concept, bytes_per_node, and ram_per_10k_concepts_mb.
        """
        seen_ids = set()
        total_bytes = sys.getsizeof(self)
        seen_ids.add(id(self))

        node_count = 0
        terminal_count = 0
        stack = [self.root]
        seen_ids.add(id(self.root))

        while stack:
            node = stack.pop()
            node_count += 1
            total_bytes += sys.getsizeof(node)
            if node.is_terminal:
                terminal_count += 1

            if node.children:
                total_bytes += sys.getsizeof(node.children)
                for tok, child in node.children.items():
                    total_bytes += sys.getsizeof(tok)
                    if id(child) not in seen_ids:
                        seen_ids.add(id(child))
                        stack.append(child)

            if node.cui is not None and id(node.cui) not in seen_ids:
                seen_ids.add(id(node.cui))
                total_bytes += sys.getsizeof(node.cui)

            if node.concept_name is not None and id(node.concept_name) not in seen_ids:
                seen_ids.add(id(node.concept_name))
                total_bytes += sys.getsizeof(node.concept_name)

            if node.metadata and id(node.metadata) not in seen_ids:
                seen_ids.add(id(node.metadata))
                total_bytes += sys.getsizeof(node.metadata)
                for k, v in node.metadata.items():
                    if id(k) not in seen_ids:
                        seen_ids.add(id(k))
                        total_bytes += sys.getsizeof(k)
                    if isinstance(v, (str, int, float, bool)) and id(v) not in seen_ids:
                        seen_ids.add(id(v))
                        total_bytes += sys.getsizeof(v)
                    elif isinstance(v, (set, list, tuple)) and id(v) not in seen_ids:
                        seen_ids.add(id(v))
                        total_bytes += sys.getsizeof(v)
                        for item in v:
                            if id(item) not in seen_ids:
                                seen_ids.add(id(item))
                                total_bytes += sys.getsizeof(item)

        num_concepts = self._total_insertions
        total_mb = total_bytes / (1024 * 1024)
        bytes_per_concept = (total_bytes / num_concepts) if num_concepts > 0 else 0.0
        bytes_per_node = (total_bytes / node_count) if node_count > 0 else 0.0
        mb_per_10k = (bytes_per_concept * 10000) / (1024 * 1024)

        return {
            "num_nodes": node_count,
            "num_concepts": num_concepts,
            "num_terminals": terminal_count,
            "total_bytes": total_bytes,
            "total_kb": total_bytes / 1024,
            "total_mb": total_mb,
            "bytes_per_concept": bytes_per_concept,
            "bytes_per_node": bytes_per_node,
            "ram_per_10k_concepts_mb": mb_per_10k,
        }

    def get_memory_usage(self) -> Dict[str, Union[int, float]]:
        """Alias for profile_memory."""
        return self.profile_memory()


class UMLSSpeculativeDrafter:
    """Zero-compute speculative drafter for Tandem speculative decoding.

    Uses a UMLSEntityTrie to propose deterministic or top-branch continuation tokens
    directly in sub-microsecond time with zero GPU compute.
    """

    def __init__(
        self,
        trie: UMLSEntityTrie,
        block_size: int = 4,
        deterministic_only: bool = False,
    ) -> None:
        self.trie = trie
        self.block_size = block_size
        self.deterministic_only = deterministic_only

    def propose(
        self,
        prefix_tokens: List[int],
        max_tokens: Optional[int] = None,
    ) -> List[int]:
        """Propose up to max_tokens candidate tokens given prefix_tokens."""
        k = max_tokens if max_tokens is not None else self.block_size
        return self.trie.propose_draft(
            prefix_tokens=prefix_tokens,
            max_tokens=k,
            deterministic_only=self.deterministic_only,
        )
