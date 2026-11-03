import os
import re
import json
import random
import time
import pandas as pd
import mock_llm

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

ZERO_SHOT_PROMPT = """CLASSIFY the following support ticket into one of these exact categories: billing, delivery, technical, fraud."""
FEW_SHOT_PROMPT = """CLASSIFY the following support ticket into one of these exact categories: billing, delivery, technical, fraud.
EXAMPLES:
- "Where is my refund for the charged order?" -> {"category": "billing", "reason": "Billing inquiry"}
- "Someone stole my account and made unauthorized purchases!" -> {"category": "fraud", "reason": "Fraudulent activity"}
- "The delivery driver never arrived with my groceries." -> {"category": "delivery", "reason": "Delivery issue"}
"""

def redact_pii(text):
    if not isinstance(text, str):
        return text, 0, 0, 0
    
    # Regex patterns
    email_pat = r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}"
    phone_pat = r"\b\d{9}\b"
    order_pat = r"ORI-\d{5}"
    
    emails = len(re.findall(email_pat, text))
    phones = len(re.findall(phone_pat, text))
    orders = len(re.findall(order_pat, text))
    
    text = re.sub(email_pat, "[EMAIL]", text)
    text = re.sub(phone_pat, "[PHONE]", text)
    text = re.sub(order_pat, "[ORDER_ID]", text)
    
    return text, emails, phones, orders

def call_with_retries(prompt, text, system_prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return mock_llm.invoke(prompt, text, system_prompt=system_prompt)
        except mock_llm.TransientLLMError:
            if attempt == max_retries - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            time.sleep(delay)

def run_pipeline(token_budget=None, prompt_type="few_shot"):
    tickets_path = os.path.join("data", "raw", "Tickets.csv")
    if not os.path.exists(tickets_path):
        print(f"Tickets file not found at {tickets_path}")
        return
        
    df = pd.read_csv(tickets_path)
    system_prompt = ZERO_SHOT_PROMPT if prompt_type == "zero_shot" else FEW_SHOT_PROMPT
    
    mock_llm.reset()
    total_tokens_used = 0
    results = []
    
    for _, row in df.iterrows():
        ticket_id = row.get("ticket_id")
        customer_id = row.get("customer_id")
        order_id = row.get("order_id")
        raw_text = str(row.get("text", ""))
        true_cat = row.get("true_category")
        
        # 1. Input Guardrails
        if not raw_text or raw_text.strip() == "":
            results.append({
                "ticket_id": ticket_id, "customer_id": customer_id, "status": "skipped_empty",
                "category": "", "segment_name": "", "priority": "", "route": "", "tokens_used": 0
            })
            continue
            
        input_tokens = mock_llm.est_tokens(raw_text)
        if input_tokens > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": ticket_id, "customer_id": customer_id, "status": "flagged_long",
                "category": "", "segment_name": "", "priority": "", "route": "", "tokens_used": input_tokens
            })
            continue
            
        # 2. Redact PII
        clean_text, _, _, _ = redact_pii(raw_text)
        
        # Check Token Budget
        if token_budget and (total_tokens_used + input_tokens > token_budget):
            results.append({
                "ticket_id": ticket_id, "customer_id": customer_id, "status": "budget_exhausted",
                "category": "", "segment_name": "", "priority": "", "route": "human_review", "tokens_used": total_tokens_used
            })
            break
            
        # 4 & 5. Resilient Call & Validation
        try:
            response = call_with_retries(prompt_type, clean_text, system_prompt)
            total_tokens_used += response.usage.get("total_tokens", 0)
            
            reply_str = response.content.strip()
            status = "flagged_invalid_output"
            category = ""
            
            try:
                parsed = json.loads(reply_str)
                cat_val = parsed.get("category", "").lower()
                if cat_val in ALLOWED_CATEGORIES:
                    category = cat_val
                    status = "classified_json"
            except Exception:
                words = re.findall(r"[a-z]+", reply_str.lower())
                found_allowed = [w for w in words if w in ALLOWED_CATEGORIES]
                if len(found_allowed) == 1:
                    category = found_allowed[0]
                    status = "classified_keyword"
                    
            route = "support_queue"
            if category == "fraud":
                route = "security_team"
            elif status in ["flagged_invalid_output", "budget_exhausted"]:
                route = "human_review"
                
            results.append({
                "ticket_id": ticket_id, "customer_id": customer_id, "status": status,
                "category": category, "segment_name": "Standard", "priority": "NORMAL", "route": route, "tokens_used": total_tokens_used
            })
            
        except Exception as e:
            results.append({
                "ticket_id": ticket_id, "customer_id": customer_id, "status": "error",
                "category": "", "segment_name": "", "priority": "", "route": "human_review", "tokens_used": total_tokens_used
            })

    os.makedirs("outputs", exist_ok=True)
    out_df = pd.DataFrame(results)
    if token_budget == 6000:
        out_df.to_csv(os.path.join("outputs", "triage_budget_demo.csv"), index=False)
        print("Saved outputs/triage_budget_demo.csv")
    else:
        out_df.to_csv(os.path.join("outputs", "triage_results.csv"), index=False)
        print("Saved outputs/triage_results.csv")

if __name__ == "__main__":
    run_pipeline(token_budget=None, prompt_type="few_shot")
    run_pipeline(token_budget=6000, prompt_type="zero_shot")