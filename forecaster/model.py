"""TimesFM model wrapper for stock forecasting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import so users without GPU can still inspect/plan
_timesfm = None


def _load_timesfm():
    global _timesfm
    if _timesfm is None:
        try:
            import timesfm  # noqa: PLC0415
            _timesfm = timesfm
        except ImportError as e:
            raise ImportError(
                "TimesFM not installed. Run: pip install timesfm[torch]\n"
                "Or follow setup at https://github.com/google-research/timesfm"
            ) from e
    return _timesfm


@dataclass
class ForecastResult:
    """Container for a TimesFM forecast."""

    ticker: str
    horizon: int
    # Shape: (horizon,)
    point_forecast: np.ndarray
    # Shape: (horizon, n_quantiles)  — e.g. [0.1, 0.2, 0.5, 0.8, 0.9]
    quantile_forecasts: np.ndarray
    quantile_levels: list[float]
    # Original (denormalised) price units
    last_known_price: float

    @property
    def lower_80(self) -> np.ndarray:
        """80% confidence lower band (10th percentile)."""
        idx = self.quantile_levels.index(0.1) if 0.1 in self.quantile_levels else 0
        return self.quantile_forecasts[:, idx]

    @property
    def upper_80(self) -> np.ndarray:
        """80% confidence upper band (90th percentile)."""
        idx = self.quantile_levels.index(0.9) if 0.9 in self.quantile_levels else -1
        return self.quantile_forecasts[:, idx]

    @property
    def median(self) -> np.ndarray:
        """Median forecast (50th percentile)."""
        idx = self.quantile_levels.index(0.5) if 0.5 in self.quantile_levels else len(self.quantile_levels) // 2
        return self.quantile_forecasts[:, idx]

    @property
    def expected_return_pct(self) -> float:
        """Expected return % from last known price to end of horizon."""
        end_price = self.point_forecast[-1]
        return (end_price - self.last_known_price) / self.last_known_price * 100

    @property
    def upside_pct(self) -> float:
        """90th-percentile upside % from last price."""
        return (self.upper_80[-1] - self.last_known_price) / self.last_known_price * 100

    @property
    def downside_pct(self) -> float:
        """10th-percentile downside % from last price."""
        return (self.lower_80[-1] - self.last_known_price) / self.last_known_price * 100


class TimesFMForecaster:
    """Wraps the Google TimesFM model for equity price forecasting.

    Attributes:
        model_variant: HuggingFace model ID.
        max_context: Maximum context tokens fed to the model.
        quantile_levels: Quantiles to generate confidence intervals.
    """

    DEFAULT_MODEL = "google/timesfm-2.5-200m-pytorch"
    QUANTILE_LEVELS = [0.1, 0.2, 0.5, 0.8, 0.9]

    def __init__(
        self,
        model_variant: str = DEFAULT_MODEL,
        max_context: int = 512,
        device: str = "cpu",
    ) -> None:
        self.model_variant = model_variant
        self.max_context = max_context
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        tfm = _load_timesfm()
        logger.info("Loading TimesFM from %s …", self.model_variant)

        # TimesFM 2.5 PyTorch variant
        self._model = tfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.model_variant,
        )
        logger.info("Model loaded.")

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        s_min: float,
        s_max: float,
        ticker: str = "UNKNOWN",
        use_log: bool = False,
    ) -> ForecastResult:
        """Run TimesFM inference on a prepared context array.

        Args:
            context: Normalised price series (output of prepare_context).
            horizon: Number of future steps to predict.
            s_min: Normalisation minimum (for inversion).
            s_max: Normalisation maximum (for inversion).
            ticker: Ticker symbol for labelling.
            use_log: Whether the context was log-transformed.

        Returns:
            ForecastResult with point and quantile forecasts in price units.
        """
        from forecaster.data import denormalise  # local import to avoid circular

        self._ensure_model()

        # TimesFM expects a list of 1-D arrays
        point_raw, quantile_raw = self._model.forecast(
            horizon=horizon,
            inputs=[context],
            quantile_levels=self.QUANTILE_LEVELS,
        )

        # Both outputs: (batch=1, horizon) and (batch=1, horizon, n_quantiles)
        point_norm = point_raw[0]  # (horizon,)
        q_norm = quantile_raw[0]   # (horizon, n_quantiles)

        # Clip to [0, 1] before denormalising (model can extrapolate)
        point_norm = np.clip(point_norm, 0.0, 1.0)
        q_norm = np.clip(q_norm, 0.0, 1.0)

        point_prices = denormalise(point_norm, s_min, s_max, use_log)
        q_prices = np.stack(
            [denormalise(q_norm[:, i], s_min, s_max, use_log) for i in range(q_norm.shape[1])],
            axis=1,
        )

        last_price = denormalise(np.array([context[-1]]), s_min, s_max, use_log)[0]

        return ForecastResult(
            ticker=ticker,
            horizon=horizon,
            point_forecast=point_prices,
            quantile_forecasts=q_prices,
            quantile_levels=self.QUANTILE_LEVELS,
            last_known_price=float(last_price),
        )
