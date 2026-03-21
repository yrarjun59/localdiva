# diva_brain.py
from ollama._client import Client, Message
import threading

class DivaBrain:
    """
    Local LLM brain using Gemma3.
    Supports streaming tokens in real-time with interrupt support.
    """
    def __init__(self, model_name: str = "gemma3", debug: bool = True):
        self.model_name = model_name
        self.debug = debug
        self._client = Client()
        self._stop_flag = False

        # System prompt
        self.system_prompt = (
            "You are Diva, a smart assistant.\n"
            "- Respond clearly and conversationally.\n"
            "- Follow instructions literally.\n"
            "- Keep responses concise."
        )

    def chat_template(self, user_input: str):
        return [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input)
        ]

    def call_llm(self, text_input: str) -> str:
        """Full response call"""
        messages = self.chat_template(text_input)
        response = self._client.chat(model=self.model_name, messages=messages)
        llm_text = response['message'].content
        if self.debug:
            print(f"[LLM RESPONSE]: {llm_text}")
        return llm_text

    def stream(self, text_input: str):
        """Token-by-token streaming generator"""
        messages = self.chat_template(text_input)
        if self.debug:
            print("[LLM STREAM START]")
        self._stop_flag = False

        for token in self._client.chat(model=self.model_name, messages=messages, stream=True):
            if self._stop_flag:
                if self.debug:
                    print("[LLM STREAM INTERRUPTED]")
                break
            content = token['message'].content
            if self.debug:
                print(f"[STREAM TOKEN]: {content}")
            yield content

        if self.debug:
            print("[LLM STREAM END]")

    def stop(self):
        """Interrupt streaming immediately"""
        self._stop_flag = True


# ---------------- DEBUG ----------------
if __name__ == "__main__":
    brain = DivaBrain(debug=True)
    try:
        for token in brain.stream("Test parallel streaming of LLM tokens."):
            print(token, end=" ", flush=True)
    except KeyboardInterrupt:
        brain.stop()