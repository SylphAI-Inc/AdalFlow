from typing import List, Tuple
import numpy as np


def confidence_interval(
    judgements: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate the confidence interval for a list of binary judgments.

    Args:
        judgements (List[float]): List of binary judgments (1/0).
        confidence (float): Confidence level (default 0.95).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    # Step 1: Calculate the mean
    mean_score = np.mean(judgements)

    # Step 2: Calculate the standard error (SE)
    standard_error = np.std(judgements, ddof=1) / np.sqrt(len(judgements))

    # Step 3: Use the Z-critical value for the confidence interval
    z_critical = 1.96  # For a 95% CI with a normal distribution

    # Step 4: Calculate the margin of error (MoE)
    margin_of_error = z_critical * standard_error

    # Step 5: Calculate the confidence interval
    # Confidence interval (clipped to [0, 1])
    lower_bound = max(0, mean_score - margin_of_error)
    upper_bound = min(1, mean_score + margin_of_error)

    return (lower_bound, upper_bound)


if __name__ == "__main__":
    # Example binary judgments (True/False as 1/0)
    judgements = [1, 1, 0, 1, 0, 1, 1]  # Convert to 1/0
    score_range = confidence_interval(judgements)
    print(score_range)
