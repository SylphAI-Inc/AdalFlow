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
    if not isinstance(judgements, list):
        raise ValueError("judgements must be a list")

    if not all(isinstance(j, (int, float)) for j in judgements):
        raise ValueError("judgements must contain only integers or floats")

    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    # Step 1: Calculate the mean
    mean_score = np.mean(judgements)

    # Step 2: Calculate the standard error (SE)
    standard_error = np.std(judgements, ddof=1) / np.sqrt(len(judgements))

    # Step 3: Use the Z-critical value for the confidence interval based on confidence level
    z_critical = np.percentile(
        np.random.normal(0, 1, 1000000), 100 * (1 - (1 - confidence) / 2)
    )

    # Step 4: Calculate the margin of error (MoE)
    margin_of_error = z_critical * standard_error

    # Step 5: Calculate the confidence interval
    # Confidence interval (clipped to [0, 1])
    lower_bound = max(0, mean_score - margin_of_error)
    upper_bound = min(1, mean_score + margin_of_error)

    return (lower_bound, upper_bound)


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Find the longest common substring between two strings.
    """
    # Create a matrix to store lengths of longest common suffixes of substrings
    # Initialize all values to 0
    m, n = len(s1), len(s2)
    lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]

    # Variable to store the length of the longest common substring
    longest_length = 0
    # Variable to store the ending index of the longest common substring in s1
    ending_index_s1 = 0

    # Build the matrix in a bottom-up manner
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
                if lcs_matrix[i][j] > longest_length:
                    longest_length = lcs_matrix[i][j]
                    ending_index_s1 = i - 1
            else:
                lcs_matrix[i][j] = 0

    # The longest common substring
    longest_common_substring = s1[
        ending_index_s1 - longest_length + 1 : ending_index_s1 + 1
    ]

    return longest_common_substring


if __name__ == "__main__":
    # Example binary judgments (True/False as 1/0)
    judgements = [1, 1, 0, 1, 0, 1, 1]  # Convert to 1/0
    score_range = confidence_interval(judgements, confidence=0.96)
    print(score_range)

    # Example longest common substring
    s1 = "abcdfghijk"
    s2 = "abedfghxyz"
    cs = longest_common_substring(s1, s2)
    print(cs)
