# llms/llama_llm.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
from llms.base_llm import BaseLLM, LLMResponse
from config import LLAMA_MODEL, OLLAMA_BASE_URL, MAX_TOKENS, TEMPERATURE


class LlamaLLM(BaseLLM):

    def __init__(self):
        super().__init__(name=LLAMA_MODEL)
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

        # Compatible toutes versions Ollama
        if hasattr(response, 'prompt_eval_count'):
            tokens_prompt     = response.prompt_eval_count or 0
            tokens_completion = response.eval_count or 0
        elif isinstance(response, dict):
            tokens_prompt     = response.get("prompt_eval_count", 0)
            tokens_completion = response.get("eval_count", 0)
        else:
            tokens_prompt     = 0
            tokens_completion = 0

        # Extraire le texte de la réponse
        if hasattr(response, 'message'):
            text = response.message.content
        elif isinstance(response, dict):
            text = response.get("message", {}).get("content", "")
        else:
            text = str(response)

        return LLMResponse(
            response=text,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_total=tokens_prompt + tokens_completion,
            cost_usd=self.build_cost(tokens_prompt, tokens_completion),
            llm_name=self.name,
        )