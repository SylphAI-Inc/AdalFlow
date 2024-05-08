"""
We just need to very basic generator that can be used to generate text from a prompt.
"""

from core.generator import Generator
from core.openai_client import OpenAIClient

from core.component import Component
from core.data_classes import DialogTurn, UserQuery, AssistantResponse
from core.memory import Memory

# TODO: make the environment variable loading more robust, and let users specify the .env path
import dotenv

dotenv.load_dotenv()


class SimpleDialog(Component):
    def __init__(self):
        super().__init__()
        model_kwargs = {"model": "gpt-3.5-turbo"}
        task_desc_str = "You are a helpful assistant."
        self.generator = Generator(
            model_client=OpenAIClient(),
            model_kwargs=model_kwargs,
            preset_prompt_kwargs={"task_desc_str": task_desc_str},
        )
        self.chat_history = Memory()
        self.generator.print_prompt()

    def chat(self) -> str:
        print("Welcome to SimpleQA. You can ask any question. Type 'exit' to end.")
        while True:
            user_input = input("You: ")
            #
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            chat_history_str = self.chat_history()
            response = self.generator(
                input=user_input, prompt_kwargs={"chat_history_str": chat_history_str}
            )
            # save the user input and response to the memory
            self.chat_history.add_dialog_turn(
                user_query=user_input, assistant_response=response
            )
            """
            From the memory management, it is  difficult to just chain them together.
            This is similar to the retrieval. This additional step is to manage the exteral db like 
            data injection. Retrieving can be chained such as we use self.chat_history() to get the chat history.
            """
            print(f"Assistant: {response}")

    # a class to have a multiple turns and take user input


if __name__ == "__main__":
    simple_qa = SimpleDialog()
    print(simple_qa)
    print(simple_qa.chat())
