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

# constrained_max_steps_12_7fb88_run_1.json
demo_only = [
    0.8263888888888888,
    0.8333333333333334,
    0.8472222222222222,
    0.8472222222222222,
    0.8472222222222222,
    0.8472222222222222,
    0.8472222222222222,
    0.8472222222222222,
    0.8472222222222222,
    0.8472222222222222,
    0.8472222222222222,
    0.8472222222222222,
    0.8472222222222222,
]
# TrecClassifierAdal/constrained_max_steps_12_5d1bf_run_1.json
text_only = [
    0.8263888888888888,
    0.8263888888888888,
    0.8333333333333334,
    0.8333333333333334,
    0.8333333333333334,
    0.8402777777777778,
    0.8402777777777778,
    0.8402777777777778,
    0.8402777777777778,
    0.8888888888888888,
    0.8888888888888888,
    0.8888888888888888,
    0.8888888888888888,
]

import matplotlib.pyplot as plt


def plot_sequential_traing(test_score_combo):
    steps = 12
    # methods = ["text_optimizer(Text-Grad 2.0)"] * steps + [
    #     "demo_optimizer (Learn-to-Reason)"
    # ] * steps
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, 13),
        test_score_combo[:12],
        marker="o",
        label="text_optimizer(Text-Grad 2.0)",
    )
    plt.plot(
        range(13, 25),
        test_score_combo[steps : 2 * steps],
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


def plot_demo_optimizer_only(demo_only):
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(demo_only) + 1),
        demo_only,
        marker="o",
        label="demo_optimizer(Learn-to-Reason)",
    )

    plt.xlabel("Steps")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Using Only Learn-to-Reason on TREC-6 Classification")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_text_optimizer_only(text_only):
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(text_only) + 1),
        text_only,
        marker="o",
        label="text_optimizer(Text-Grad 2.0)",
    )

    plt.xlabel("Steps")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Using Only Text-Grad 2.0 on TREC-6 Classification")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    # plot_sequential_traing(test_score_combo)
    # plot_demo_optimizer_only(demo_only)  # Training time: 95.81865906715393s
    plot_text_optimizer_only(text_only)  # Training time: 95.81865906715393s
