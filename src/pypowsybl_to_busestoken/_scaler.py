# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""BusesTokenScaler: physics-informed feature normalisation for ML training."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd

from ._token import BusesToken


# ---------------------------------------------------------------------------
# Stateless transform helpers
# ---------------------------------------------------------------------------

def _signed_log1p(s: pd.Series) -> pd.Series:
    """
    Symmetric log1p transform: sign(x) * log1p(|x|).

    Properties:
    - Preserves sign and zero: f(0) = 0, f(-x) = -f(x).
    - Compresses large magnitudes: 100 → 4.62, 1000 → 6.91.
    - No NaN introduced for any finite input.
    - Handles the zero-heavy power injection distributions (gen_p, p_net, …)
      without requiring a global shift or scale parameter.

    Used for: gen_p, gen_q, load_p, load_q, shunt_q, bat_p, bat_q,
              p_net, q_net, p1, q1, p2, q2, r, x (handles rare negatives).
    """
    return np.sign(s) * np.log1p(s.abs())


def _log1p_positive(s: pd.Series) -> pd.Series:
    """
    log1p for strictly non-negative quantities.  NaN is preserved.

    Used for: i1, i2 (current in A), limit1, limit2 (IST limits in A),
              n_gens, n_loads, n_shunts, n_batteries (counts).
    """
    return np.log1p(s)


def _log10_positive(s: pd.Series, *, eps: float = 1e-9) -> pd.Series:
    """
    log10 for strictly positive quantities, with a floor ``eps`` for zeros.
    NaN is preserved.

    Used for: nominal_v (kV), b1, b2 (line charging susceptance — small positive).
    The floor eps=1e-9 maps zero-charging lines to -9 (distinct from small but
    non-zero values starting around -5 for 63 kV lines).
    """
    return np.log10(s.clip(lower=eps))


def _log10_tap_rho(s: pd.Series) -> pd.Series:
    """
    log10(tap_rho) for the off-nominal transformation ratio.

    - Lines : tap_rho = 1.0 exactly → log10(1) = 0.
    - 2WT   : tap_rho in (0.015, 1.78] → log10 in (-1.83, +0.25].
      Step-down 400→63 kV ≈ 0.16 → log10 ≈ -0.80.
      Step-down 400→90 kV ≈ 0.23 → log10 ≈ -0.64.

    This centering at 0 for lines is natural: the transformer ratio is the
    deviation from the "pass-through" case.
    """
    return np.log10(s)


def _v_mag_pu(bus_df: pd.DataFrame) -> pd.Series:
    """
    Convert v_mag (kV) to per-unit: v_mag / nominal_v.

    After this transform the distribution is approximately N(1.03, 0.028²)
    across all voltage levels, removing the multi-modal kV structure.
    Common transmission voltage levels each give values close to 1 p.u.
    """
    return bus_df["v_mag"] / bus_df["nominal_v"]


# ---------------------------------------------------------------------------
# Fitted transforms (require statistics computed over a training set)
# ---------------------------------------------------------------------------

class _ZScoreTransform:
    """Store and apply a per-column z-score: (x - μ) / σ."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = float(mean)
        self.std = float(std)

    def fit(self, s: pd.Series) -> "_ZScoreTransform":
        self.mean = float(s.mean())
        self.std = float(s.std())
        return self

    def transform(self, s: pd.Series) -> pd.Series:
        return (s - self.mean) / max(self.std, 1e-9)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "_ZScoreTransform":
        return cls(mean=d["mean"], std=d["std"])


class _RobustScaler:
    """
    (x - median) / IQR, optionally clipped before scaling.

    Clip is applied first so that extreme outliers (e.g. base_rho=9.03 from
    artificial limits) do not distort the IQR estimate.
    Fitted on non-NaN values; NaN is preserved after transform.
    """

    def __init__(
        self,
        q25: float = 0.0,
        q75: float = 1.0,
        clip: float | None = None,
    ) -> None:
        self.q25 = float(q25)
        self.q75 = float(q75)
        self.clip = clip  # upper clip applied before scaling (None = no clip)

    @property
    def _iqr(self) -> float:
        return max(self.q75 - self.q25, 1e-9)

    def fit(self, s: pd.Series, clip: float | None = None) -> "_RobustScaler":
        """
        Fit on non-NaN values.  If ``clip`` is provided it is stored and used
        during transform (clip is applied *before* quantile computation to
        avoid outlier contamination of the quantile estimates).
        """
        self.clip = clip
        vals = s.dropna()
        if clip is not None:
            vals = vals.clip(upper=clip)
        self.q25 = float(np.percentile(vals, 25))
        self.q75 = float(np.percentile(vals, 75))
        return self

    def transform(self, s: pd.Series) -> pd.Series:
        out = s.copy()
        if self.clip is not None:
            out = out.clip(upper=self.clip)
        return (out - self.q25) / self._iqr

    def to_dict(self) -> dict:
        return {"q25": self.q25, "q75": self.q75, "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "_RobustScaler":
        return cls(q25=d["q25"], q75=d["q75"], clip=d.get("clip"))


# ---------------------------------------------------------------------------
# Main scaler class
# ---------------------------------------------------------------------------

class BusesTokenScaler:
    """
    Physics-informed feature normaliser for :class:`BusesToken`.

    Transforms both ``bus_df`` (token features) and ``relation_df`` (relation
    features) in-place on copies, returning a new :class:`BusesToken` with
    normalised values.

    Design principles
    -----------------
    * **Stateless where possible**: most transforms use only per-row physics
      (e.g. ``v_mag / nominal_v``) or fixed mathematical functions
      (``signed_log1p``, ``log10``).
    * **Fitted only for two quantities**:
      - ``v_angle``: z-score  (mean and std vary with grid region/time).
      - ``base_rho``: RobustScaler with upper clip  (outliers from artificial
        limits inflate the maximum dramatically).
    * **Sign conventions are preserved**: ``gen_p > 0`` = producing,
      ``load_p > 0`` = consuming — ``signed_log1p`` keeps the sign.
    * **NaN is preserved**: missing limits (``limit1``, ``limit2``,
      ``base_rho``) and transformer shunt columns (``g1``, ``b1`` for 2WT)
      remain NaN after transform, so the model can mask them if needed.

    Token (bus_df) transforms
    -------------------------
    =====================  ==========================================
    Feature                Transform
    =====================  ==========================================
    ``v_mag``              ``v_mag / nominal_v`` → p.u.  (≈ N(1.03, 0.028²))
    ``v_angle``            z-score  (fitted; μ≈−6.78°, σ≈15.41°)
    ``nominal_v``          ``log10(nominal_v)`` → continuous voltage embedding
    ``is_main_component``  identity (binary 0/1)
    ``gen_p``, ``gen_q``   ``signed_log1p``
    ``load_p``, ``load_q`` ``signed_log1p``
    ``shunt_q``            ``signed_log1p``
    ``bat_p``, ``bat_q``   ``signed_log1p``
    ``p_net``, ``q_net``   ``signed_log1p``
    ``n_gens``, …          ``log1p(n)``
    =====================  ==========================================

    Relation (relation_df) transforms
    ----------------------------------
    ========================  =============================================
    Feature                   Transform
    ========================  =============================================
    ``r``, ``x``              ``signed_log1p``  (handles rare negative r/x)
    ``g1``, ``g2``            identity  (always 0 for lines; NaN for 2WT)
    ``b1``, ``b2``            ``log10(|b| + 1e-9)`` — line charging
                              (0 → −9, typical → −5…−3)
    ``tap_rho``               ``log10(tap_rho)`` (lines → 0; 2WT → −1.8…+0.25)
    ``tap_alpha``             identity  (0 for lines/ratio-tap; non-zero PSTs)
    ``p1``, ``q1``, ``p2``,   ``signed_log1p``
    ``q2``
    ``i1``, ``i2``            ``log1p(i)``  (always positive, in A)
    ``limit1``, ``limit2``    ``log1p(limit)``  (positive, NaN preserved)
    ``base_rho``              ``RobustScaler`` with clip  (fitted)
    ``is_line``, …            identity  (binary flags)
    ========================  =============================================

    Usage
    -----
    >>> scaler = BusesTokenScaler()
    >>> scaler.fit(list_of_busestoken_snapshots)
    >>> normalised = scaler.transform(token)

    Or in one shot:

    >>> normalised_list = scaler.fit_transform(list_of_busestoken_snapshots)

    Serialisation
    -------------
    >>> scaler.to_json("scaler.json")
    >>> scaler2 = BusesTokenScaler.from_json("scaler.json")
    """

    # Default clip for base_rho (> p99 = 0.58; set conservatively at 2.0
    # to allow slight overloads while clipping artificial-limit artefacts).
    _BASE_RHO_CLIP: float = 2.0

    def __init__(self) -> None:
        self._fitted = False
        self._angle_scaler = _ZScoreTransform()
        self._base_rho_scaler = _RobustScaler(clip=self._BASE_RHO_CLIP)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, snapshots: Iterable[BusesToken]) -> "BusesTokenScaler":
        """
        Compute statistics for fitted transforms from a collection of snapshots.

        Fitted transforms:
        - ``v_angle`` z-score (mean, std over all buses in all snapshots).
        - ``base_rho`` RobustScaler (Q25, Q75 over all branches, after clipping
          at ``_BASE_RHO_CLIP``).

        Parameters
        ----------
        snapshots :
            Iterable of :class:`BusesToken` objects (training set).
            Can be a list or a generator.

        Returns
        -------
        self
        """
        angle_vals: list[np.ndarray] = []
        base_rho_vals: list[np.ndarray] = []

        for tok in snapshots:
            angle_vals.append(tok.bus_df["v_angle"].dropna().values)
            br = tok.relation_df["base_rho"].dropna().values
            base_rho_vals.append(br[br <= self._BASE_RHO_CLIP])

        angles = np.concatenate(angle_vals) if angle_vals else np.array([0.0])
        base_rho_parts = [a for a in base_rho_vals if len(a) > 0]
        base_rhos = (
            np.concatenate(base_rho_parts)
            if base_rho_parts
            else np.array([0.0, 0.25])  # fallback when no limits defined (e.g. IEEE test nets)
        )

        self._angle_scaler.mean = float(angles.mean())
        self._angle_scaler.std = float(angles.std())

        self._base_rho_scaler.q25 = float(np.percentile(base_rhos, 25))
        self._base_rho_scaler.q75 = float(np.percentile(base_rhos, 75))
        self._base_rho_scaler.clip = self._BASE_RHO_CLIP

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, token: BusesToken) -> BusesToken:
        """
        Return a new :class:`BusesToken` with all features normalised.

        The original ``token`` is not modified.  ``snapshot_id`` is preserved.
        Columns not listed in the transform spec (metadata: voltage_level_id,
        substation_id, connected_component, bus1_id, bus2_id, branch_kind)
        are carried through unchanged.

        Parameters
        ----------
        token : BusesToken
            Must have been produced by :class:`BusesTokenConverter`.
        """
        if not self._fitted:
            raise RuntimeError(
                "Scaler not fitted — call .fit() or .fit_transform() first."
            )

        bus_df = self._transform_bus(token.bus_df.copy())
        relation_df = self._transform_relation(token.relation_df.copy())
        return BusesToken(
            bus_df=bus_df,
            relation_df=relation_df,
            snapshot_id=token.snapshot_id,
        )

    def fit_transform(
        self, snapshots: list[BusesToken]
    ) -> list[BusesToken]:
        """
        Fit on ``snapshots`` then transform each one.

        Parameters
        ----------
        snapshots : list of BusesToken

        Returns
        -------
        list of BusesToken (normalised, same order as input)
        """
        self.fit(snapshots)
        return [self.transform(t) for t in snapshots]

    # ------------------------------------------------------------------
    # Internal per-dataframe transforms
    # ------------------------------------------------------------------

    def _transform_bus(self, df: pd.DataFrame) -> pd.DataFrame:
        # v_mag → p.u. (division by nominal_v)
        if "v_mag" in df.columns and "nominal_v" in df.columns:
            df["v_mag"] = _v_mag_pu(df)

        # v_angle → z-score
        if "v_angle" in df.columns:
            df["v_angle"] = self._angle_scaler.transform(df["v_angle"])

        # nominal_v → log10 (continuous voltage-level embedding)
        if "nominal_v" in df.columns:
            df["nominal_v"] = _log10_positive(df["nominal_v"])

        # Power injections: signed_log1p
        _slog1p_cols = [
            "gen_p", "gen_q",
            "load_p", "load_q",
            "shunt_q",
            "bat_p", "bat_q",
            "p_net", "q_net",
        ]
        for col in _slog1p_cols:
            if col in df.columns:
                df[col] = _signed_log1p(df[col])

        # Device counts: log1p
        for col in ["n_gens", "n_loads", "n_shunts", "n_batteries"]:
            if col in df.columns:
                df[col] = _log1p_positive(df[col].astype(float))

        # is_main_component → identity (already 0/1)

        return df

    def _transform_relation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Series impedance r, x: signed_log1p (handles rare negative values)
        for col in ["r", "x"]:
            if col in df.columns:
                df[col] = _signed_log1p(df[col])

        # Shunt conductance g1, g2: always 0 for lines, NaN for 2WT → identity
        # (no-op; column kept as-is so schema is preserved)

        # Line charging susceptance b1, b2: log10 with small-value floor
        for col in ["b1", "b2"]:
            if col in df.columns:
                df[col] = _log10_positive(df[col], eps=1e-9)

        # Tap ratio: log10(tap_rho). Lines → 0. 2WT step-down → negative.
        if "tap_rho" in df.columns:
            df["tap_rho"] = _log10_tap_rho(df["tap_rho"])

        # Phase shift: identity (usually 0, non-zero only for PSTs)

        # Power flows p1, q1, p2, q2: signed_log1p
        for col in ["p1", "q1", "p2", "q2"]:
            if col in df.columns:
                df[col] = _signed_log1p(df[col])

        # Currents i1, i2: always positive → log1p
        for col in ["i1", "i2"]:
            if col in df.columns:
                df[col] = _log1p_positive(df[col])

        # IST limits: log1p (positive; NaN preserved)
        for col in ["limit1", "limit2"]:
            if col in df.columns:
                df[col] = _log1p_positive(df[col])

        # Loading ratio base_rho: RobustScaler with clip
        if "base_rho" in df.columns:
            df["base_rho"] = self._base_rho_scaler.transform(df["base_rho"])

        # Binary flags: is_line, is_2wt, is_self_loop → identity

        return df

    # ------------------------------------------------------------------
    # Fitted statistics (for inspection / logging)
    # ------------------------------------------------------------------

    @property
    def angle_mean(self) -> float:
        """Fitted mean of v_angle (degrees)."""
        return self._angle_scaler.mean

    @property
    def angle_std(self) -> float:
        """Fitted std of v_angle (degrees)."""
        return self._angle_scaler.std

    @property
    def base_rho_q25(self) -> float:
        """Fitted Q25 of base_rho (used in RobustScaler denominator)."""
        return self._base_rho_scaler.q25

    @property
    def base_rho_q75(self) -> float:
        """Fitted Q75 of base_rho (used in RobustScaler denominator)."""
        return self._base_rho_scaler.q75

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise fitted statistics to a plain dict (JSON-compatible)."""
        return {
            "fitted": self._fitted,
            "base_rho_clip": self._BASE_RHO_CLIP,
            "angle_scaler": self._angle_scaler.to_dict(),
            "base_rho_scaler": self._base_rho_scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BusesTokenScaler":
        """Restore a :class:`BusesTokenScaler` from a serialised dict."""
        scaler = cls()
        scaler._fitted = d.get("fitted", False)
        scaler._angle_scaler = _ZScoreTransform.from_dict(d["angle_scaler"])
        scaler._base_rho_scaler = _RobustScaler.from_dict(d["base_rho_scaler"])
        return scaler

    def to_json(self, path: str) -> None:
        """Save fitted statistics to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "BusesTokenScaler":
        """Load a :class:`BusesTokenScaler` from a JSON file."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __repr__(self) -> str:
        if not self._fitted:
            return "BusesTokenScaler(not fitted)"
        return (
            f"BusesTokenScaler("
            f"angle_mean={self.angle_mean:.3f}°, "
            f"angle_std={self.angle_std:.3f}°, "
            f"base_rho_q25={self.base_rho_q25:.4f}, "
            f"base_rho_q75={self.base_rho_q75:.4f}, "
            f"clip={self._BASE_RHO_CLIP})"
        )
