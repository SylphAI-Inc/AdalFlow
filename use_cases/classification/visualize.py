# constrained_max_steps_12_848d2_run_7.json
test_score_combo = (
    [
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
    ],
)
import matplotlib.pyplot as plt

methods = ["text_optimizer"] * 12 + ["demo_optimizer"] * 12
plt.figure(figsize=(10, 6))
plt.plot(range(1, 13), test_score_combo[:12], marker="o", label="text_optimizer")
plt.plot(range(13, 25), test_score_combo[12:24], marker="o", label="demo_optimizer")

plt.axvline(x=12.5, color="gray", linestyle="--")  # Divider between methods

plt.xlabel("Steps")
plt.ylabel("Test Score")
plt.title("Test Score by Optimization Method")
plt.legend()
plt.grid(True)

plt.show()
