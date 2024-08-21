# constrained_max_steps_12_848d2_run_7.json
test_score_combo = [
    0.8263888888888888,
    0.8263888888888888,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8611111111111112,
    0.8819444444444444,
    0.8819444444444444,
    0.8819444444444444,
    0.8819444444444444,
    0.8819444444444444,
    0.8819444444444444,
    0.8819444444444444,
    0.8819444444444444,
    0.8819444444444444,
    0.8819444444444444,
]
import matplotlib.pyplot as plt

methods = ["text_optimizer(Text-Grad 2.0)"] * 12 + [
    "demo_optimizer (Learn-to-Reason)"
] * 12
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, 13),
    test_score_combo[:12],
    marker="o",
    label="text_optimizer(Text-Grad 2.0)",
)
plt.plot(
    range(13, 25),
    test_score_combo[12:24],
    marker="o",
    label="demo_optimizer(Learn-to-Reason)",
)

plt.axvline(x=12.5, color="gray", linestyle="--")  # Divider between methods

plt.xlabel("Steps")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Using Seqential Optimization on TREC-6 Classification")
plt.legend()
plt.grid(True)

plt.show()
