import random
import re

class TransientLLMError(Exception):
    """Custom exception simulating transient LLM API errors."""
    pass

class MockResponse:
    def __init__(self, content="", tool_calls=None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage or {"total_tokens": 0}

class MockLLM:
    def __init__(self):
        self.call_count = 0
        self.error_trigger_count = 0

    def reset(self):
        self.call_count = 0
        self.error_trigger_count = 0

    def est_tokens(self, text):
        if not text:
            return 0
        if isinstance(text, list):
            text = " ".join([str(m) for m in text])
        return len(str(text).split()) + int(len(str(text)) / 4)

    def invoke(self, prompt, text, system_prompt=""):
        self.call_count += 1
        if self.call_count % 8 == 0 and self.error_trigger_count < 19:
            self.error_trigger_count += 1
            raise TransientLLMError("Simulated transient network timeout.")

        t_lower = str(text).lower()
        if "bill" in t_lower or "charge" in t_lower or "refund" in t_lower or "payment" in t_lower:
            cat = "billing"
        elif "deliver" in t_lower or "driver" in t_lower or "late" in t_lower or "order" in t_lower:
            cat = "delivery"
        elif "fraud" in t_lower or "scam" in t_lower or "stolen" in t_lower or "hack" in t_lower:
            cat = "fraud"
        else:
            cat = "technical"

        response_text = f'{{"category": "{cat}", "reason": "The customer appears to have a {cat} related problem."}}'
        total_tokens = self.est_tokens(system_prompt) + self.est_tokens(text) + self.est_tokens(response_text)
        return MockResponse(content=response_text, usage={"total_tokens": total_tokens})

    def chat(self, messages, tools=None):
        self.call_count += 1
        msg_str = " ".join([str(m) for m in messages])
        tokens = self.est_tokens(msg_str) + 50

        # Check if caller is asking standard policy RAG question
        for m in reversed(messages):
            content = str(m.get("content", ""))
            if "CONTEXT:" in content:
                if "could not find" in content.lower() or "unanswerable" in content.lower() or "missing" in content.lower():
                    return MockResponse(content="I could not find the information in the provided policy documents.", usage={"total_tokens": tokens})
                return MockResponse(content="Based on the policy context provided, refunds are processed according to store rules.", usage={"total_tokens": tokens})

        # Tool calling simulation for ReAct Agent (Scenarios S1 - S7)
        history = [str(m) for m in messages]
        full_text = " ".join(history)

        # Check existing tool executions in message history
        tool_executed = any(m.get("role") == "tool" for m in messages)
        last_tool_name = ""
        last_tool_res = {}
        for m in reversed(messages):
            if m.get("role") == "tool":
                last_tool_name = m.get("name")
                try:
                    import json
                    last_tool_res = json.loads(m.get("content", "{}"))
                except:
                    last_tool_res = {}
                break

        # Decision tree based on scenarios
        if "S1" in full_text or "ORD-1001" in full_text:
            if not any(m.get("role") == "tool" and m.get("name") == "get_order" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_1", "name": "get_order", "arguments": {"order_id": "ORD-1001"}}], usage={"total_tokens": tokens})
            elif not any(m.get("role") == "tool" and m.get("name") == "search_policy" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_2", "name": "search_policy", "arguments": {"query": "refund policy late delivery"}}], usage={"total_tokens": tokens})
            elif not any(m.get("role") == "tool" and m.get("name") == "calculate_refund" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_3", "name": "calculate_refund", "arguments": {"order_amount": 128.25, "delivery_minutes": 52}}], usage={"total_tokens": tokens})
            elif not any(m.get("role") == "tool" and m.get("name") == "issue_refund" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_4", "name": "issue_refund", "arguments": {"order_id": "ORD-1001", "refund_inr": 128.25}}], usage={"total_tokens": tokens})
            else:
                return MockResponse(content="REFUND processed successfully for order ORD-1001 in the amount of 128.25.", usage={"total_tokens": tokens})

        elif "S2" in full_text or "ORD-1002" in full_text:
            if not any(m.get("role") == "tool" and m.get("name") == "get_order" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_1", "name": "get_order", "arguments": {"order_id": "ORD-1002"}}], usage={"total_tokens": tokens})
            elif not any(m.get("role") == "tool" and m.get("name") == "calculate_refund" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_2", "name": "calculate_refund", "arguments": {"order_amount": 135.00, "delivery_minutes": 60}}], usage={"total_tokens": tokens})
            elif not any(m.get("role") == "tool" and m.get("name") == "issue_refund" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_3", "name": "issue_refund", "arguments": {"order_id": "ORD-1002", "refund_inr": 135.00}}], usage={"total_tokens": tokens})
            else:
                return MockResponse(content="REFUND issued for ORD-1002.", usage={"total_tokens": tokens})

        elif "S3" in full_text or "ORD-1003" in full_text:
            if not any(m.get("role") == "tool" and m.get("name") == "get_order" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_1", "name": "get_order", "arguments": {"order_id": "ORD-1003"}}], usage={"total_tokens": tokens})
            elif not any(m.get("role") == "tool" and m.get("name") == "calculate_refund" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_2", "name": "calculate_refund", "arguments": {"order_amount": 1999.00, "delivery_minutes": 55}}], usage={"total_tokens": tokens})
            elif not any(m.get("role") == "tool" and m.get("name") == "escalate_to_human" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_3", "name": "escalate_to_human", "arguments": {"order_id": "ORD-1003", "refund_inr": 1999.00}}], usage={"total_tokens": tokens})
            else:
                return MockResponse(content="ESCALATE order ORD-1003 to human support agent.", usage={"total_tokens": tokens})

        elif "S4" in full_text or "ORD-1004" in full_text:
            if not any(m.get("role") == "tool" and m.get("name") == "get_order" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_1", "name": "get_order", "arguments": {"order_id": "ORD-1004"}}], usage={"total_tokens": tokens})
            elif not any(m.get("role") == "tool" and m.get("name") == "calculate_refund" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_2", "name": "calculate_refund", "arguments": {"order_amount": 80.00, "delivery_minutes": 30}}], usage={"total_tokens": tokens})
            else:
                return MockResponse(content="DENIED refund because order was delivered within 45 minutes.", usage={"total_tokens": tokens})

        elif "S5" in full_text or "ORD-1005" in full_text:
            if not any(m.get("role") == "tool" and m.get("name") == "get_order" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_1", "name": "get_order", "arguments": {"order_id": "ORD-1005"}}], usage={"total_tokens": tokens})
            elif not any(m.get("role") == "tool" and m.get("name") == "calculate_refund" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_2", "name": "calculate_refund", "arguments": {"order_amount": 95.00, "delivery_minutes": 25}}], usage={"total_tokens": tokens})
            else:
                return MockResponse(content="DENIED refund since order delivered under 45 mins.", usage={"total_tokens": tokens})

        elif "S6" in full_text or "ORD-9999" in full_text:
            if not any(m.get("role") == "tool" and m.get("name") == "get_order" for m in messages):
                return MockResponse(tool_calls=[{"id": "call_1", "name": "get_order", "arguments": {"order_id": "ORD-9999"}}], usage={"total_tokens": tokens})
            else:
                return MockResponse(content="NOT_FOUND: Order ORD-9999 does not exist.", usage={"total_tokens": tokens})

        elif "S7" in full_text or "ORD-1007" in full_text:
            return MockResponse(tool_calls=[{"id": "call_loop", "name": "search_policy", "arguments": {"query": "refund policy"}}], usage={"total_tokens": tokens})

        # Generic response fallback
        return MockResponse(content="Request processed.", usage={"total_tokens": tokens})

_llm = MockLLM()

def reset():
    _llm.reset()

def est_tokens(text):
    return _llm.est_tokens(text)

def invoke(prompt, text, system_prompt=""):
    return _llm.invoke(prompt, text, system_prompt=system_prompt)

def chat(messages, tools=None):
    return _llm.chat(messages, tools=tools)