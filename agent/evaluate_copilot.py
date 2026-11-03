import os
import pandas as pd
from copilot import InstamartCopilot

SCENARIOS_PATH = os.path.join("data", "raw", "agent_scenarios.csv")
OUTPUT_PATH = os.path.join("outputs", "agent_evaluation_results.csv")

os.makedirs("outputs", exist_ok=True)

def run_evaluation():
    print("Loading agent scenarios...")
    if not os.path.exists(SCENARIOS_PATH):
        print(f"Scenarios file not found at {SCENARIOS_PATH}")
        return

    scenarios = pd.read_csv(SCENARIOS_PATH)
    copilot = InstamartCopilot()

    results = []
    for _, row in scenarios.iterrows():
        customer_id = row.get("customer_id")
        scenario_id = row.get("scenario_id")
        
        # Get copilot recommendation
        rec = copilot.recommend_action(customer_id)
        
        results.append({
            "scenario_id": scenario_id,
            "customer_id": customer_id,
            "customer_tier": rec.get("customer_tier") if isinstance(rec, dict) else "N/A",
            "lapse_probability": rec.get("lapse_probability") if isinstance(rec, dict) else 0.0,
            "recommended_action": rec.get("recommended_action") if isinstance(rec, dict) else rec
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Evaluation completed! Results saved to {OUTPUT_PATH}")
    print(results_df.head())

if __name__ == "__main__":
    run_evaluation()