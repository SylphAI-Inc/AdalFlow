from typing import Dict
from typing import List


class Data:
    """
    The basic data classes used for LightRAG. History is a list of dictionaries, each containing a query and an answer.
    """

    def __init__(
        self, query: str = "", history: List[Dict[str, str]] = None, answer: str = ""
    ):
        self.query = query
        self.history = history if history is not None else []
        self.answer = answer

    def add_to_history(self, query: str, answer: str):
        """
        Adds a query and its corresponding answer to the history.
        """
        self.history.append({"query": query, "answer": answer})

    def get_history(self) -> List[Dict[str, str]]:
        """
        Returns the dialog history.
        """
        return self.history

    def clear_history(self):
        """
        Clears the history.
        """
        self.history.clear()

    def get_history_length(self) -> int:
        """
        Returns the number of query-answer pairs in the history.
        """
        return len(self.history)

    def get_last_query(self) -> str:
        """
        Returns the last query from the history.
        """
        return self.history[-1]["query"] if self.history else None

    def get_last_answer(self) -> str:
        """
        Returns the last answer from the history.
        """
        return self.history[-1]["answer"] if self.history else None

    def get_query_answer_pair(self, index: int) -> Dict[str, str]:
        """
        Returns the query-answer pair at a specific index.
        """
        return self.history[index] if 0 <= index < len(self.history) else None


if __name__ == "__main__":
    data = Data()
    data.add_to_history("What's your name?", "My name is LightRAG.")
    data.add_to_history("How are you?", "I'm fine, thank you.")
    print(
        data.get_history()
    )  # Outputs: [{'query': "What's your name?", 'answer': 'My name is AI.'}, {'query': 'How are you?', 'answer': "I'm fine, thank you."}]
