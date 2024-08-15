"""
We just need to very basic generator that can be used to generate text from a prompt.
"""

# from adalflow.core.component import Component
# from adalflow.core.memory import Memory

# from adalflow.components.model_client import OpenAIClient


# class SimpleDialog(Component):
#     def __init__(self):
#         super().__init__()
#         model_kwargs = {"model": "gpt-3.5-turbo"}
#         task_desc_str = "You are a helpful assistant."
#         self.generator = Generator(
#             model_client=OpenAIClient(),
#             model_kwargs=model_kwargs,
#             preset_prompt_kwargs={"task_desc_str": task_desc_str},
#         )
#         self.chat_history = Memory()
#         self.generator.print_prompt()

#     def chat(self) -> str:
#         print("Welcome to SimpleQA. You can ask any question. Type 'exit' to end.")
#         while True:
#             user_input = input("You: ")
#             #
#             if user_input.lower() == "exit":
#                 print("Goodbye!")
#                 break
#             chat_history_str = self.chat_history()
#             response = self.generator(
#                 prompt_kwargs={
#                     "chat_history_str": chat_history_str,
#                     "input": user_input,
#                 },
#             )
#             # save the user input and response to the memory
#             self.chat_history.add_dialog_turn(
#                 user_query=user_input, assistant_response=response
#             )
#             """
#             Most components mush have a __call__ method in order to be chained together with other component in the data pipeline.
#             From the memory management, it is  difficult to just chain them together.
#             This is similar to the retrieval. This additional step is to manage the exteral db like
#             data injection. Retrieving can be chained such as we use self.chat_history() to get the chat history.
#             """
#             print(f"Assistant: {response}")

#     # a class to have a multiple turns and take user input


# if __name__ == "__main__":
#     simple_qa = SimpleDialog()
#     print(simple_qa)
#     print(simple_qa.chat())
