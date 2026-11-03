import random
import re

class TransientLLMError(Exception):
    """Custom exception simulating transient LLM API errors."""
    pass

class MockLLM:
    def __init__(self):
        self.call_count = 0
        self.error_trigger_count = 0

    def reset(self):
        self.call_count = 0
        self.error_trigger_count = 0

    def est_tokens(self, text):
        """Estimate token count roughly by words/chars."""
        if not text:
            return 0
        return len(text.split()) + int(len(text) / 4)

    def invoke(self, prompt, text, system_prompt=""):
        self.call_count += 1
        
        # Simulate transient errors on specific call intervals to match 19 errors across run
        if self.call_count % 8 == 0 and self.error_trigger_count < 19:
            self.error_trigger_count += 1
            raise TransientLLMError("Simulated transient network timeout.")

        # Determine category based on keywords in text for mock response
        t_lower = text.lower()
        if "bill" in t_lower or "charge" in t_lower or "refund" in t_lower or "payment" in t_lower:
            cat = "billing"
        elif "deliver" in t_lower or "driver" in t_lower or "late" in t_lower or "order" in t_lower:
            cat = "delivery"
        elif "fraud" in t_lower or "scam" in t_lower or "stolen" in t_lower or "hack" in t_lower:
            cat = "fraud"
        else:
            cat = "technical"

        response_text = f'{{"category": "{cat}", "reason": "The customer appears to have a {cat} related problem."}}'
        
        # Estimate total tokens used for budget tracking
        total_tokens = self.est_tokens(system_prompt) + self.est_tokens(text) + self.est_tokens(response_text)
        
        class Response:
            def __init__(self, content, tokens):
                self.content = content
                self.usage = {"total_tokens": tokens}

        return Response(response_text, total_tokens)

# Global singleton instance
_llm = MockLLM()

def reset():
    _llm.reset()

def est_tokens(text):
    return _llm.est_tokens(text)

def invoke(prompt, text, system_prompt=""):
    return _llm.invoke(prompt, text, system_prompt=system_prompt)