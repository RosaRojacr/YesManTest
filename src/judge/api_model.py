import os
import anthropic

PROJECT_ROOT = "R:\\YesManTest"


class APIModel:
    def __init__(self, model_name="claude-haiku-4-5-20251001", api_key_path=None):
        if api_key_path is None:
            api_key_path = os.path.join(PROJECT_ROOT, "Authentication", "Anthropic_Key.txt")
        with open(api_key_path) as f:
            api_key = f.read().strip()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def ask(self, system_prompt, user_message, max_new_tokens=256):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.content[0].text