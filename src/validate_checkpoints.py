"""src/validate_checkpoints.py - Asserts the reference checkpoints for Part A
Must print exactly: ALL CHECKPOINTS PASSED
"""

import os
import sqlite3
import pandas as pd
import numpy as np

def validate_all():
    db_path = os.path.join("data", "instamart.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_path} does not exist. Run Part A first.")

    conn = sqlite3.connect(db_path)
    cust = pd.read_sql_query("SELECT * FROM customers", conn)
    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    conn.close()

    # Checkpoint 1: Clean customer city counts
    expected_cities = {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197,
        "Chennai": 191, "Hyderabad": 191, "Mumbai": 184
    }
    actual_cities = cust["city"].value_counts().to_dict()
    assert actual_cities == expected_cities, f"City counts mismatch: {actual_cities} != {expected_cities}"

    # Checkpoint 2: Clean customer gender counts
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    actual_genders = cust["gender"].value_counts().to_dict()
    assert actual_genders == expected_genders, f"Gender counts mismatch: {actual_genders} != {expected_genders}"

    # Checkpoint 3: Missing age count is 0
    assert cust["age"].isna().sum() == 0, "There are missing age values in customers."

    # Checkpoint 4: Customer ages between 18 and 60
    assert cust["age"].min() >= 18 and cust["age"].max() <= 60, f"Customer age bounds violated: min={cust['age'].min()}, max={cust['age'].max()}"

    # Checkpoint 5: Clean orders row count is 12,309 and all order_ids are distinct
    assert len(orders) == 12309, f"Clean orders count mismatch: {len(orders)} != 12309"
    assert orders["order_id"].nunique() == 12309, "Duplicate order_id found in clean orders."

    # Checkpoint 6: Status counts (11,110 completed, 1,199 cancelled, 0 pending)
    status_counts = orders["status"].value_counts().to_dict()
    assert status_counts.get("completed", 0) == 11110, f"Completed count mismatch: {status_counts.get('completed')}"
    assert status_counts.get("cancelled", 0) == 1199, f"Cancelled count mismatch: {status_counts.get('cancelled')}"
    assert "pending" not in status_counts or status_counts["pending"] == 0, "Pending orders found in clean orders."

    # Checkpoint 7: 0 missing order_amount
    assert orders["order_amount"].isna().sum() == 0, "Missing order_amount found."

    # Checkpoint 8: 0 missing delivery_minutes for completed orders
    comp = orders[orders["status"] == "completed"]
    assert comp["delivery_minutes"].isna().sum() == 0, "Missing delivery_minutes in completed orders."

    # Checkpoint 9: Exactly 1,199 missing delivery_minutes for cancelled orders
    canc = orders[orders["status"] == "cancelled"]
    assert canc["delivery_minutes"].isna().sum() == 1199, "Cancelled orders should have missing delivery_minutes."

    # Checkpoint 10: Sum of completed order_amount is exactly INR 2,998,668.80
    completed_revenue = round(float(comp["order_amount"].sum()), 2)
    assert np.isclose(completed_revenue, 2998668.80), f"Completed revenue mismatch: {completed_revenue} != 2998668.80"

    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate_all()
