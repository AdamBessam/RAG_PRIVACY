# llms/mistral_llm.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
from llms.base_llm import BaseLLM, LLMResponse
from config import MISTRAL_MODEL, OLLAMA_BASE_URL, MAX_TOKENS, TEMPERATURE


class MistralLLM(BaseLLM):
    """
    Mistral 7B v0.3 via Ollama local.
    """

    def __init__(self):
        super().__init__(name=MISTRAL_MODEL)
        self.client = ollama.Client(host=OLLAMA_BASE_URL)

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.name,
            messages=messages,
            options={
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
            }
        )

        tokens_prompt     = response.prompt_eval_count or 0
        tokens_completion = response.eval_count or 0

        return LLMResponse(
            response=response.message.content,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_total=tokens_prompt + tokens_completion,
            cost_usd=self.build_cost(tokens_prompt, tokens_completion),
            llm_name=self.name,
        )