from ollama import chat
from ollama import ChatResponse


class OllamaChat:
    def __init__(self, system_prompt: str = None, max_history: int = -1):
        self.messages = []
        self.system_prompt = system_prompt
        self.max_history = max_history
        if self.system_prompt:
            self.messages.append({ 'role': 'system', 'content': self.system_prompt })

    def chat(self, user_message: str, temperature:float = 0.7, format=None) -> str:
        if self.max_history > 0 and len(self.messages) >= self.max_history * 2 + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history * 2):]

        self.messages.append({ 'role': 'user', 'content': user_message })

        stream: ChatResponse = chat(model='ikiru/Dolphin-Mistral-24B-Venice-Edition',
                                      messages=self.messages, options={'temperature': temperature},
                                      keep_alive=True, stream=True, format=format)

        generated_text = ""
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            generated_text += chunk['message']['content']
        print()

        self.messages.append({ 'role': 'assistant', 'content': generated_text })
        return generated_text

    def clear_history(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({'role': 'system', 'content': self.system_prompt})

    def get_latest_response(self) -> str:
        for message in reversed(self.messages):
            if message['role'] == 'assistant':
                return message['content']
        return ""




