training_time = (2005 * 3 + 3056) / 4
test_scores = [0.84, 0.86, 0.79, 0.89]
val_scores = [0.95, 0.97, 0.93, 0.96]

import numpy as np

mean_test_score = np.mean(test_scores)
std_test_score = np.std(test_scores)

# val scores
mean_val_score = np.mean(val_scores)
std_val_score = np.std(val_scores)

print(f"Mean test score: {mean_test_score}")
print(f"Standard deviation of test scores: {std_test_score}")
print(f"Mean validation score: {mean_val_score}")
print(f"Standard deviation of validation scores: {std_val_score}")
print(f"Average training time: {training_time}")

# Mean test score: 0.8450000000000001
# Standard deviation of test scores: 0.036400549446402586
# Mean validation score: 0.9525
# Standard deviation of validation scores: 0.014790199457749011

val_accs = [
    [
        0.74,
        0.89,
        0.92,
        0.92,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
    ],
    [
        0.74,
        0.92,
        0.92,
        0.92,
        0.92,
        0.92,
        0.92,
        0.92,
        0.92,
        0.92,
        0.92,
        0.92,
        0.92,
        0.97,
    ],
    [0.74, 0.87, 0.87, 0.88, 0.88, 0.9, 0.9, 0.9, 0.9, 0.9, 0.91, 0.91, 0.93, 0.93],
    [
        0.74,
        0.85,
        0.91,
        0.91,
        0.91,
        0.91,
        0.91,
        0.91,
        0.91,
        0.91,
        0.91,
        0.91,
        0.96,
        0.96,
    ],
]

pass_rates = [len(set(val_acc)) / len(val_acc) for val_acc in val_accs]
average_pass_rate = np.mean(pass_rates)

print(f"Average pass rate: {average_pass_rate}")  # 30.35
