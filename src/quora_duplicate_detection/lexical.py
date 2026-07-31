"""A portable lexical classifier with no pickle-based serialization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from quora_duplicate_detection.features import FEATURE_NAMES, PairFeatureExtractor


@dataclass
class LexicalModel:
    extractor: PairFeatureExtractor
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        ngram_range: tuple[int, int],
        max_features: int,
        min_df: int,
        logistic_c: float,
        max_iter: int,
        random_seed: int,
    ) -> LexicalModel:
        extractor = PairFeatureExtractor(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
        )
        features = extractor.fit_transform(frame)
        labels = frame["is_duplicate"].to_numpy(dtype=np.int8)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        classifier = LogisticRegression(
            C=logistic_c,
            max_iter=max_iter,
            random_state=random_seed,
            solver="lbfgs",
        )
        classifier.fit(scaled, labels)
        return cls(
            extractor=extractor,
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
            coefficients=np.asarray(classifier.coef_[0], dtype=np.float64),
            intercept=float(classifier.intercept_[0]),
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.extractor.transform(frame)
        scaled = (features - self.scaler_mean) / self.scaler_scale
        return expit(scaled @ self.coefficients + self.intercept)

    def validate(self) -> None:
        expected = (len(FEATURE_NAMES),)
        for name, array in (
            ("scaler_mean", self.scaler_mean),
            ("scaler_scale", self.scaler_scale),
            ("coefficients", self.coefficients),
        ):
            if np.asarray(array).shape != expected:
                raise ValueError(f"{name} has an invalid shape")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if (self.scaler_scale <= 0).any():
            raise ValueError("scaler_scale must be positive")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept is not finite")
