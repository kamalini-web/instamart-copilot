import os
import sqlite3
import pandas as pd
import numpy as np

# Ensure directories exist
os.makedirs(os.path.join("data", "clean"), exist_ok=True)
DB_PATH = os.path.join("data", "instamart.db")

print("Loading raw data...")
customers = pd.read_csv(os.path.join("data", "raw", "customers.csv"))
products = pd.read_csv(os.path.join("data", "raw", "products.csv"))
orders = pd.read_csv(os.path.join("data", "raw", "orders.csv"))
tickets = pd.read_csv(os.path.join("data", "raw", "tickets.csv"))

# --- 1. CLEAN CUSTOMERS ---
customers = customers.drop_duplicates()
customers["city"] = customers["city"].str.strip().str.title()
customers["gender"] = customers["gender"].str.strip().str.lower()
gender_map = {"m": "male", "f": "female", "male": "male", "female": "female"}
customers["gender"] = customers["gender"].map(gender_map).fillna("other")

# --- 2. CLEAN PRODUCTS ---
products = products.drop_duplicates()
products["category"] = products["category"].str.strip()

# --- 3. CLEAN ORDERS ---
orders = orders.drop_duplicates()
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders = orders.sort_values(by=["customer_id", "order_ts"]).reset_index(drop=True)

# Fill missing / incorrect order_amount using product price, quantity, and discount
merged_orders = orders.merge(products, on="product_id", how="left")
calc_amount = np.round(merged_orders["price_inr"] * merged_orders["quantity"] * (1 - merged_orders["discount_pct"] / 100), 2)
orders["order_amount"] = orders["order_amount"].fillna(calc_amount)

# Fix delivery minutes outliers / missing values for completed orders
completed_mask_merged = merged_orders["status"] == "completed"
cat_median_deliv = merged_orders[completed_mask_merged].groupby("category")["delivery_minutes"].median()

for cat, median_val in cat_median_deliv.items():
    cat_pids = products[products["category"] == cat]["product_id"]
    sub_mask = orders["product_id"].isin(cat_pids) & (orders["status"] == "completed") & (orders["delivery_minutes"].isna() | (orders["delivery_minutes"] < 0))
    orders.loc[sub_mask, "delivery_minutes"] = median_val

# --- 4. SAVE CLEANED CSVs ---
customers.to_csv(os.path.join("data", "clean", "customers.csv"), index=False)
products.to_csv(os.path.join("data", "clean", "products.csv"), index=False)
orders.to_csv(os.path.join("data", "clean", "orders.csv"), index=False)
tickets.to_csv(os.path.join("data", "clean", "tickets.csv"), index=False)
print("Cleaned CSVs saved successfully.")

# --- 5. STORE IN SQLITE DATABASE ---
conn = sqlite3.connect(DB_PATH)
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()
print(f"Database created and tables loaded successfully at {DB_PATH}.")