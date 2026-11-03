import os
import sqlite3
import pandas as pd
import joblib

DB_PATH = os.path.join("data", "instamart.db")
MODEL_PATH = os.path.join("models", "lapse_model.joblib")

class InstamartCopilot:
    def __init__(self, db_path=DB_PATH, model_path=MODEL_PATH):
        self.db_path = db_path
        self.model = joblib.load(model_path) if os.path.exists(model_path) else None

    def get_customer_profile(self, customer_id):
        """Fetch customer profile, segment tier, and lapse prediction."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM customer_segments WHERE customer_id = ?"
        profile = pd.read_sql(query, conn, params=(customer_id,))
        conn.close()
        
        if profile.empty:
            return f"Customer {customer_id} not found."
        return profile.to_dict(orient="records")[0]

    def recommend_action(self, customer_id):
        """Generate tailored copilot recommendations based on tier & lapse risk."""
        profile = self.get_customer_profile(customer_id)
        if isinstance(profile, str):
            return profile
        
        tier = profile.get("customer_tier", "Regular")
        lapse_prob = profile.get("lapse_probability", 0.0)
        tickets = profile.get("support_tickets", 0)

        # Recommendation logic
        if lapse_prob > 0.6:
            action = f"HIGH CHURN RISK ({lapse_prob:.0%}): Issue targeted 15% discount coupon or loyalty perk."
        elif "VIP" in tier:
            action = f"VIP CUSTOMER: Priority routing & offer proactive support resolution."
        elif tickets > 2:
            action = f"FREQUENT TICKETS ({tickets}): Assign to senior support agent for follow-up."
        else:
            action = f"STANDARD: Provide standard support assistance."

        return {
            "customer_id": customer_id,
            "customer_tier": tier,
            "lapse_probability": round(lapse_prob, 2),
            "recommended_action": action
        }

if __name__ == "__main__":
    copilot = InstamartCopilot()
    # Test sample customer lookup
    sample_res = copilot.recommend_action("C0001")
    print("Copilot Test Output:")
    print(sample_res)