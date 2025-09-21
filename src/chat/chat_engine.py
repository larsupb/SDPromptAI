from abc import abstractmethod, ABC


class IChatEngine(ABC):
    @abstractmethod
    def chat(self, user_message: str, max_new_tokens: int = 512, do_sample: bool = True, temperature: float = 0.7,
             top_k: int = 50, top_p: float = 1.0, repetition_penalty: float = 1.1) -> str:
        """
        Generates a chat response based on the user message and parameters.
        The generated message is stored in the chat history and returned.
        :param user_message: The message from the user to respond to.
        :param max_new_tokens: The maximum number of tokens to generate in the response.
        :param temperature: Controls the randomness of the response. Higher values yield more random results.
        :param top_k: Limits the next token selection to the top K tokens with the highest probabilities.
        :param top_p: Limits the next token selection to a subset of tokens with a cumulative probability of top_p.
        :param repetition_penalty: Penalizes repeated tokens to reduce redundancy in the response.
        :return: The generated chat response as a string.
        """
        pass

    @abstractmethod
    def get_latest_response(self) -> str:
        pass