"""Reward functions for Challenger and Reasoner."""

import math
from typing import List
import numpy as np
from .verifier import verify, AnswerType


def reasoner_reward(
    y_pred: str,
    y_gold: str,
    ans_type: AnswerType,
    mcq_options: List[str] | None = None
) -> float:
    """Compute Reasoner reward (0.0 or 1.0 based on correctness)."""
    is_correct = verify(y_pred, y_gold, ans_type, mcq_options)
    return 1.0 if is_correct else 0.0


def challenger_reward(
    correctness_list: List[float],
    sigma: float = 0.15
) -> tuple[float, float]:
    """Compute Gaussian reward for Challenger (peaks at 50% pass-rate)."""
    if len(correctness_list) == 0:
        return 0.0, 0.0
    
    p = float(np.mean(correctness_list))
    exponent = -((p - 0.5) ** 2) / (2 * (sigma ** 2))
    reward = math.exp(exponent)
    reward = max(0.0, min(1.0, reward))
    
    return reward, p

