import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Ensure output folders exist
os.makedirs("models", exist_ok=True)
DB_PATH = os.path.join("data", "instamart.db")

print("Connecting to database and loading data...")
conn = sqlite3.connect(DB_PATH)
customers = pd.read_sql("SELECT * FROM customers", conn)
orders = pd.read_sql("SELECT * FROM orders WHERE status = 'completed'", conn)
conn.close()

# Load tickets from CSV
tickets = pd.read_csv(os.path.join("data", "clean", "tickets.csv"))

# Ensure numeric columns are properly converted
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

# Convert timestamp to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
reference_date = orders["order_ts"].max() + pd.Timedelta(days=1)

print("Engineering features (RFM & Behavioral)...")
# 1. RFM Features
rfm = orders.groupby("customer_id").agg(
    recency_days=("order_ts", lambda x: (reference_date - x.max()).days),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum"),
    avg_order_value=("order_amount", "mean"),
    avg_delivery_mins=("delivery_minutes", "mean")
).reset_index()

# 2. Ticket / Support Features
ticket_counts = tickets.groupby("customer_id").size().reset_index(name="support_tickets")
customer_features = customers.merge(rfm, on="customer_id", how="left").merge(ticket_counts, on="customer_id", how="left")

# Fill missing values for customers with no orders/tickets
customer_features["recency_days"] = customer_features["recency_days"].fillna(customer_features["recency_days"].max())
customer_features["frequency"] = customer_features["frequency"].fillna(0)
customer_features["monetary"] = customer_features["monetary"].fillna(0)
customer_features["avg_order_value"] = customer_features["avg_order_value"].fillna(0)
customer_features["avg_delivery_mins"] = customer_features["avg_delivery_mins"].fillna(customer_features["avg_delivery_mins"].mean())
customer_features["support_tickets"] = customer_features["support_tickets"].fillna(0)

# --- PART 1: CUSTOMER SEGMENTATION (K-Means Clustering) ---
print("Running K-Means Customer Segmentation...")
seg_features = customer_features[["recency_days", "frequency", "monetary", "support_tickets"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(seg_features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_features["cluster"] = kmeans.fit_predict(scaled_features)

# Map clusters to meaningful tier names based on monetary/frequency
cluster_summary = customer_features.groupby("cluster")[["monetary", "frequency"]].mean()
vip_cluster = cluster_summary["monetary"].idxmax()
at_risk_cluster = cluster_summary["monetary"].idxmin()

def map_tier(c):
    if c == vip_cluster:
        return "VIP / High-Spender"
    elif c == at_risk_cluster:
        return "At-Risk / Inactive"
    else:
        return "Regular"

customer_features["customer_tier"] = customer_features["cluster"].apply(map_tier)


# --- PART 2: LAPSE PREDICTION MODEL (Random Forest) ---
print("Training Lapse Prediction Model...")
customer_features["is_lapsed"] = (customer_features["recency_days"] > 60).astype(int)

X = customer_features[["frequency", "monetary", "avg_order_value", "avg_delivery_mins", "support_tickets"]]
y = customer_features["is_lapsed"]

if y.nunique() > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    customer_features["lapse_probability"] = model.predict_proba(X)[:, 1]
    joblib.dump(model, os.path.join("models", "lapse_model.joblib"))
    print("Model trained and saved to models/lapse_model.joblib")
else:
    customer_features["lapse_probability"] = 0.0
    print("Skipping train_test_split due to single class distribution.")

# --- PART 3: STORE RESULTS IN SQLITE ---
print("Saving customer segments and predictions to SQLite database...")
conn = sqlite3.connect(DB_PATH)
customer_features[[
    "customer_id", "customer_tier", "recency_days", "frequency", 
    "monetary", "support_tickets", "is_lapsed", "lapse_probability"
]].to_sql("customer_segments", conn, if_exists="replace", index=False)
conn.close()

print("Part B execution completed successfully! Tables saved in instamart.db.")