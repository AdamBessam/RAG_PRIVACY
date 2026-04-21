# llms/gpt4o_mini_llm.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from llms.base_llm import BaseLLM, LLMResponse
from config import OPENAI_API_KEY, GPT4O_MINI_MODEL, MAX_TOKENS, TEMPERATURE


class GPT4oMiniLLM(BaseLLM):

    def __init__(self):
        super().__init__(name="gpt4o-mini")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model  = GPT4O_MINI_MODEL

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        text              = response.choices[0].message.content or ""
        tokens_prompt     = response.usage.prompt_tokens
        tokens_completion = response.usage.completion_tokens
        tokens_total      = response.usage.total_tokens
        cost              = self.build_cost(tokens_prompt, tokens_completion)

        return LLMResponse(
            response=text,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_total=tokens_total,
            cost_usd=cost,
            llm_name=self.name,
        )
