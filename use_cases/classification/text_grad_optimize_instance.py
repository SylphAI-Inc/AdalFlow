import textgrad as tg
from adalflow.utils import setup_env

setup_env()

tg.set_backward_engine("gpt-4o", override=True)

# Step 1: Get an initial response from an LLM.
model = tg.BlackboxLLM("gpt-4o")
question_string = (
    "If it takes 1 hour to dry 25 shirts under the sun, "
    "how long will it take to dry 30 shirts under the sun? "
    "Reason step by step"
)

question = tg.Variable(
    question_string, role_description="question to the LLM", requires_grad=False
)

answer = model(question)
print(answer)

answer.set_role_description("concise and accurate answer to the question")

# Step 2: Define the loss function and the optimizer, just like in PyTorch!
# Here, we don't have SGD, but we have TGD (Textual Gradient Descent)
# that works with "textual gradients".
optimizer = tg.TGD(parameters=[answer])
evaluation_instruction = (
    f"Here's a question: {question_string}. "
    "Evaluate any given answer to this question, "
    "be smart, logical, and very critical. "
    "Just provide concise feedback."
)


# TextLoss is a natural-language specified loss function that describes
# how we want to evaluate the reasoning.
loss_fn = tg.TextLoss(evaluation_instruction)

# Step 3: Do the loss computation, backward pass, and update the punchline.
# Exact same syntax as PyTorch!
loss = loss_fn(answer)
loss.backward()
optimizer.step()
print(answer)
