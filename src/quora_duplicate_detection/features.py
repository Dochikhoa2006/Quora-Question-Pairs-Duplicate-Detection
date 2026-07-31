"""Symmetric, null-safe lexical features for question pairs."""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from quora_duplicate_detection.data import PAIR_COLUMNS, normalize_question

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)

FEATURE_NAMES = (
    "character_length_ratio",
    "token_length_ratio",
    "normalized_character_length_difference",
    "normalized_token_length_difference",
    "dice_overlap",
    "jaccard_similarity",
    "token_containment",
    "sequence_similarity",
    "token_sort_similarity",
    "numeric_token_overlap",
    "first_token_agreement",
    "last_token_agreement",
    "exact_normalized_match",
    "character_ngram_tfidf_cosine",
)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _length_ratio(left: int, right: int) -> float:
    largest = max(left, right)
    return 1.0 if largest == 0 else min(left, right) / largest


def _normalized_difference(left: int, right: int) -> float:
    largest = max(left, right)
    return 0.0 if largest == 0 else abs(left - right) / largest


def _set_overlap(left: set[str], right: set[str]) -> tuple[float, float, float]:
    if not left or not right:
        return 0.0, 0.0, 0.0
    intersection_size = len(left & right)
    dice = 2.0 * intersection_size / (len(left) + len(right))
    jaccard = intersection_size / len(left | right)
    containment = intersection_size / min(len(left), len(right))
    return dice, jaccard, containment


def _numeric_overlap(left: set[str], right: set[str]) -> float:
    left_numbers = {token for token in left if token.isdecimal()}
    right_numbers = {token for token in right if token.isdecimal()}
    if not left_numbers or not right_numbers:
        return 0.0
    return 2.0 * len(left_numbers & right_numbers) / (len(left_numbers) + len(right_numbers))


def _sequence_similarity(left: str, right: str) -> float:
    """Average both directions because SequenceMatcher tie-breaking is directional."""
    forward = SequenceMatcher(None, left, right, autojunk=False).ratio()
    backward = SequenceMatcher(None, right, left, autojunk=False).ratio()
    return (forward + backward) / 2.0


class PairFeatureExtractor:
    """Fit a corpus TF-IDF model and generate the documented 14 features."""

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (3, 5),
        max_features: int = 50_000,
        min_df: int = 2,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.effective_min_df = min_df
        self.vectorizer: TfidfVectorizer | None = None

    def _new_vectorizer(
        self,
        *,
        min_df: int,
        vocabulary: dict[str, int] | None = None,
    ) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            lowercase=False,
            norm="l2",
            dtype=np.float64,
            max_features=self.max_features,
            min_df=min_df,
            vocabulary=vocabulary,
        )

    def fit(self, frame: pd.DataFrame) -> PairFeatureExtractor:
        questions = self._normalized_questions(frame)
        corpus = [*questions[0], *questions[1]]
        self.vectorizer = self._new_vectorizer(min_df=self.min_df)
        try:
            self.vectorizer.fit(corpus)
        except ValueError:
            # Tiny test/demo corpora may have no terms at min_df > 1.
            self.effective_min_df = 1
            self.vectorizer = self._new_vectorizer(min_df=1)
            self.vectorizer.fit(corpus if any(corpus) else ["empty"])
        return self

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("feature extractor is not fitted")
        question1, question2 = self._normalized_questions(frame)
        tokens1 = [tokenize(text) for text in question1]
        tokens2 = [tokenize(text) for text in question2]

        tfidf1 = self.vectorizer.transform(question1)
        tfidf2 = self.vectorizer.transform(question2)
        tfidf_cosine = np.asarray(tfidf1.multiply(tfidf2).sum(axis=1)).ravel()

        rows = [
            self._pair_features(left, right, left_tokens, right_tokens, tfidf_score)
            for left, right, left_tokens, right_tokens, tfidf_score in zip(
                question1,
                question2,
                tokens1,
                tokens2,
                tfidf_cosine,
                strict=True,
            )
        ]
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.shape != (len(frame), len(FEATURE_NAMES)):
            raise RuntimeError("feature matrix has an unexpected shape")
        if not np.isfinite(matrix).all():
            raise RuntimeError("feature matrix contains non-finite values")
        return matrix

    @staticmethod
    def _normalized_questions(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        missing = sorted(set(PAIR_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        return (
            [normalize_question(value) for value in frame["question1"]],
            [normalize_question(value) for value in frame["question2"]],
        )

    @staticmethod
    def _pair_features(
        left: str,
        right: str,
        left_tokens: list[str],
        right_tokens: list[str],
        tfidf_cosine: float,
    ) -> list[float]:
        left_set = set(left_tokens)
        right_set = set(right_tokens)
        dice, jaccard, containment = _set_overlap(left_set, right_set)
        token_sorted_left = " ".join(sorted(left_tokens))
        token_sorted_right = " ".join(sorted(right_tokens))

        return [
            _length_ratio(len(left), len(right)),
            _length_ratio(len(left_tokens), len(right_tokens)),
            _normalized_difference(len(left), len(right)),
            _normalized_difference(len(left_tokens), len(right_tokens)),
            dice,
            jaccard,
            containment,
            _sequence_similarity(left, right),
            _sequence_similarity(token_sorted_left, token_sorted_right),
            _numeric_overlap(left_set, right_set),
            float(bool(left_tokens and right_tokens and left_tokens[0] == right_tokens[0])),
            float(bool(left_tokens and right_tokens and left_tokens[-1] == right_tokens[-1])),
            float(bool(left) and left == right),
            float(tfidf_cosine),
        ]

    @property
    def vocabulary(self) -> dict[str, int]:
        if self.vectorizer is None:
            raise RuntimeError("feature extractor is not fitted")
        return {term: int(index) for term, index in self.vectorizer.vocabulary_.items()}

    @property
    def idf(self) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("feature extractor is not fitted")
        return np.asarray(self.vectorizer.idf_, dtype=np.float64)

    @classmethod
    def from_state(
        cls,
        *,
        vocabulary: dict[str, int],
        idf: Iterable[float],
        ngram_range: tuple[int, int],
        max_features: int,
        min_df: int,
    ) -> PairFeatureExtractor:
        extractor = cls(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
        )
        extractor.effective_min_df = min_df
        extractor.vectorizer = extractor._new_vectorizer(
            min_df=min_df,
            vocabulary=vocabulary,
        )
        idf_array = np.asarray(list(idf), dtype=np.float64)
        if idf_array.shape != (len(vocabulary),):
            raise ValueError("TF-IDF IDF vector does not match the vocabulary")
        extractor.vectorizer.idf_ = idf_array
        return extractor
