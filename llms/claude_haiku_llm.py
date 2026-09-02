# llms/claude_haiku_llm.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from llms.base_llm import BaseLLM, LLMResponse
from config import ANTHROPIC_API_KEY, CLAUDE_HAIKU_MODEL, MAX_TOKENS, TEMPERATURE


class ClaudeHaikuLLM(BaseLLM):

    def __init__(self):
        super().__init__(name="claude-haiku")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model  = CLAUDE_HAIKU_MODEL

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        kwargs = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        text              = response.content[0].text if response.content else ""
        tokens_prompt     = response.usage.input_tokens
        tokens_completion = response.usage.output_tokens
        tokens_total      = tokens_prompt + tokens_completion
        cost              = self.build_cost(tokens_prompt, tokens_completion)

        return LLMResponse(
            response=text,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_total=tokens_total,
            cost_usd=cost,
            llm_name=self.name,
        )
