# Instamart Customer Intelligence & Support Copilot

## Overview
This capstone project implements an end-to-end customer intelligence and support automation copilot for Instamart. It integrates data engineering SQL pipelines, customer segmentation and churn prediction machine learning models, guardrailed ticket triage workflows, policy RAG retrieval, and an autonomous ReAct refund agent with safety limits and checkpointing.

## Reproduce
To replicate the entire pipeline and generate all model outputs, ledgers, and evaluation reports, run the following scripts in order:
1. `python src/clean_data.py`
2. `python src/train_predict.py`
3. `python triage.py`
4. `python part_d_rag_agent.py`

## Part A: Data Engineering & SQL
- Cleaned raw data, generated clean CSVs, built the SQLite database (`instamart.db`), and wrote analytical queries in `sql/queries.sql`.

## Part B: Machine Learning
- Engineered RFM and behavioral features, trained a Random Forest classification model (`models/lapse_model.joblib`), and stored churn/lapse predictions.

## Part C: Support Ticket Triage with Guardrails
- Implemented input guardrails (skipping empty tickets, flagging long tickets >400 tokens), PII redaction (emails, phone numbers, order IDs), zero-shot and few-shot classification prompts, resilient exponential backoff retries for `TransientLLMError`, and output validation/routing. Results saved to `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.

## Part D: Policy RAG Assistant and Refund Agent
- Built a TF-IDF policy document retriever with 40-word chunks and 10-word overlaps.
- Implemented an autonomous ReAct agent loop equipped with 5 tools (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`) interacting with `instamart.db` and logging to `outputs/refund_ledger.jsonl` and `outputs/agent_runs.jsonl`.
- Enforced safety fallback limits (max iterations, token limit, timeout) and state checkpointing in `state/` for resumable runs.

## Limitations and Next Steps
- **Limitations:** TF-IDF retrieval relies on lexical matching rather than deep semantic embeddings; mock LLM simulation handles expected test vectors.
- **Next Steps:** Integrate production vector databases (e.g., FAISS/Pinecone) with dense embedding models and connect live LLM APIs.