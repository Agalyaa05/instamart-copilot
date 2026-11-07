"""src/part_a.py - Data Foundation: Clean, Store, Query
Audits raw data, standardises customers and orders, persists clean tables into SQLite
and CSVs, and answers analytical business questions via SQL and pandas.
"""

import os
import sqlite3
import pandas as pd
import numpy as np

def run_part_a():
    os.makedirs(os.path.join("data", "clean"), exist_ok=True)
    raw_dir = os.path.join("data", "raw")
    
    # 1. Load Raw Datasets
    cust_raw = pd.read_csv(os.path.join(raw_dir, "customers.csv"))
    prod_raw = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    ord_raw = pd.read_csv(os.path.join(raw_dir, "orders.csv"))
    tix_raw = pd.read_csv(os.path.join(raw_dir, "tickets.csv"), keep_default_na=False)

    print("==================================================================")
    print("PART A: DATA FOUNDATION AUDIT & CLEANING")
    print("==================================================================")

    # 2. Audit Raw Data
    print("\n--- Raw Table Shapes & Types ---")
    for name, df in [("Customers", cust_raw), ("Products", prod_raw), ("Orders", ord_raw), ("Tickets", tix_raw)]:
        print(f"{name}: shape={df.shape}")
        missing_s = df.isna().sum()
        pct_missing = (missing_s / len(df)) * 100
        audit_df = pd.DataFrame({"Missing Count": missing_s, "Missing %": pct_missing.round(2)})
        print(audit_df)
        print("-" * 40)

    # Specific Audit Metrics
    raw_orders_rows = len(ord_raw)
    exact_duplicates = int(ord_raw.duplicated().sum())
    pending_rows = int((ord_raw["status"] == "pending").sum())
    distinct_orders = int(ord_raw["order_id"].nunique())
    
    amount_numeric = pd.to_numeric(ord_raw["order_amount"], errors="coerce")
    non_numeric_amounts = int(amount_numeric.isna().sum())
    
    deliv_numeric = pd.to_numeric(ord_raw["delivery_minutes"], errors="coerce")
    negative_delivery = int((deliv_numeric < 0).sum())
    blank_delivery = int(ord_raw["delivery_minutes"].isna().sum())

    blank_age = int(cust_raw["age"].isna().sum())
    cust_age_num = pd.to_numeric(cust_raw["age"], errors="coerce")
    outside_age = int(((cust_age_num < 10) | (cust_age_num > 100)).sum())
    raw_cities_count = int(cust_raw["city"].nunique())
    
    valid_gender_series = cust_raw["gender"].dropna().loc[lambda s: s.astype(str).str.strip() != ""]
    distinct_genders_count = int(valid_gender_series.nunique())
    blank_gender = int(cust_raw["gender"].isna().sum() + (cust_raw["gender"].astype(str).str.strip() == "").sum())

    print("\n--- Specific Raw Audit Checkpoints ---")
    audit_summary = {
        "Raw orders rows": raw_orders_rows,
        "Exact duplicate rows": exact_duplicates,
        "Pending rows": pending_rows,
        "Distinct order_id": distinct_orders,
        "Non-numeric amounts": non_numeric_amounts,
        "Negative delivery times": negative_delivery,
        "Blank delivery times": blank_delivery,
        "Blank age": blank_age,
        "Age outside 10-100": outside_age,
        "Distinct raw cities": raw_cities_count,
        "Distinct raw genders (non-blank)": distinct_genders_count,
        "Blank gender": blank_gender
    }
    for k, v in audit_summary.items():
        print(f"  {k:35}: {v}")

    # 3. Clean Customers
    cust = cust_raw.copy()
    # (a) Standardise city: strip and title-case
    cust["city"] = cust["city"].astype(str).str.strip().str.title()
    
    # (b) Map gender to male, female, unknown
    def clean_gender(g):
        if pd.isna(g) or str(g).strip() == "":
            return "unknown"
        s = str(g).strip().lower()
        if s in ["male", "m"]:
            return "male"
        elif s in ["female", "f"]:
            return "female"
        return "unknown"

    cust["gender"] = cust["gender"].apply(clean_gender)
    
    # (c) Treat ages outside 10-100 as missing, fill missing with city median age
    cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
    cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
    cust["age"] = cust.groupby("city")["age"].transform(lambda s: s.fillna(s.median()))
    
    # (d) Convert signup_date to datetime
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])

    print("\n--- Customers Cleaned Checkpoints ---")
    print("City counts:\n", cust["city"].value_counts().to_dict())
    print("Gender counts:\n", cust["gender"].value_counts().to_dict())
    print(f"Missing age: {cust['age'].isna().sum()}, Min age: {cust['age'].min()}, Max age: {cust['age'].max()}")

    # 4. Clean Orders
    orders = ord_raw.copy()
    # Step 1: Convert order_ts to datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    
    # Step 2: Remove exact duplicate rows
    orders = orders.drop_duplicates()
    
    # Step 3: Keep only the record with latest order_ts per order_id
    orders = orders.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last").sort_index()

    # Merge product price and category for recalculations and imputation
    orders = orders.merge(prod_raw[["product_id", "category", "price_inr"]], on="product_id", how="left")
    
    # Step 4: Convert order_amount to numeric, impute missing from product catalog price
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    calc_amount = (orders["price_inr"] * orders["quantity"] * (1.0 - orders["discount_pct"] / 100.0)).round(2)
    orders["order_amount"] = orders["order_amount"].fillna(calc_amount)

    # Step 5: Treat negative delivery_minutes as missing
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    
    # Impute missing delivery_minutes for completed orders using product category median
    completed_mask = orders["status"] == "completed"
    orders.loc[completed_mask, "delivery_minutes"] = (
        orders[completed_mask].groupby("category")["delivery_minutes"].transform(lambda s: s.fillna(s.median()))
    )
    # Cancelled orders remain missing
    orders.loc[orders["status"] == "cancelled", "delivery_minutes"] = np.nan

    # Format order_ts back to string for database and clean files
    orders_clean = orders[["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]].copy()
    orders_clean["order_ts"] = orders_clean["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    cust_clean = cust.copy()
    cust_clean["signup_date"] = cust_clean["signup_date"].dt.strftime("%Y-%m-%d")

    prod_clean = prod_raw.copy()

    print("\n--- Orders Cleaned Checkpoints ---")
    print(f"Total clean orders: {len(orders_clean)}")
    print("Status counts:\n", orders_clean["status"].value_counts().to_dict())
    print(f"Missing order_amount: {orders_clean['order_amount'].isna().sum()}")
    comp_orders = orders_clean[orders_clean["status"] == "completed"]
    canc_orders = orders_clean[orders_clean["status"] == "cancelled"]
    print(f"Completed missing delivery: {comp_orders['delivery_minutes'].isna().sum()}")
    print(f"Cancelled missing delivery: {canc_orders['delivery_minutes'].isna().sum()}")
    print(f"Sum completed revenue: INR {comp_orders['order_amount'].sum():.2f}")

    # 5. Store to CSV and SQLite
    clean_dir = os.path.join("data", "clean")
    cust_clean.to_csv(os.path.join(clean_dir, "customers.csv"), index=False)
    prod_clean.to_csv(os.path.join(clean_dir, "products.csv"), index=False)
    orders_clean.to_csv(os.path.join(clean_dir, "orders.csv"), index=False)

    db_path = os.path.join("data", "instamart.db")
    conn = sqlite3.connect(db_path)
    cust_clean.to_sql("customers", conn, if_exists="replace", index=False)
    prod_clean.to_sql("products", conn, if_exists="replace", index=False)
    orders_clean.to_sql("orders", conn, if_exists="replace", index=False)
    print(f"\nPersisted clean tables to {clean_dir} and SQLite DB at {db_path}")

    # 6. Execute SQL Queries
    print("\n==================================================================")
    print("SQL ANALYSIS (sql/queries.sql)")
    print("==================================================================")
    
    with open(os.path.join("sql", "queries.sql"), "r", encoding="utf-8") as f:
        sql_content = f.read()

    clean_lines = [line for line in sql_content.splitlines() if not line.strip().startswith("--")]
    clean_sql = "\n".join(clean_lines)
    queries_to_run = [q.strip() for q in clean_sql.split(";") if q.strip()]

    query_names = [
        "Q1: WHERE then HAVING (>500 amount, >8000 total)",
        "Q2: Three-table join (Revenue by city and category)",
        "Q3: Scalar subquery (Products priced above average)",
        "Q4: LEFT JOIN (Customers with no completed orders)",
        "Q5: Common Table Expression (Customers above avg spend)",
        "Q6: Ranking window functions (RANK, DENSE_RANK, ROW_NUMBER)",
        "Q7: Running total (Daily and cumulative revenue)"
    ]

    for name, query in zip(query_names, queries_to_run):
        df_res = pd.read_sql_query(query, conn)
        print(f"\n--- {name} ---")
        print(f"Result rows: {len(df_res)}")
        if "Q1" in name:
            print("Row count:", len(df_res))
        elif "Q2" in name:
            print("Total Revenue: INR", round(df_res["revenue"].sum(), 2), "Top cell:", df_res.iloc[0].to_dict())
        elif "Q3" in name:
            print("Products above average:", len(df_res))
        elif "Q4" in name:
            print("Customers with no completed order:", len(df_res))
        elif "Q5" in name:
            avg_val = pd.read_sql_query("WITH cs AS (SELECT customer_id, SUM(order_amount) AS ts FROM orders WHERE status = 'completed' GROUP BY customer_id) SELECT AVG(ts) FROM cs;", conn).iloc[0, 0]
            print(f"Qualifying customers: {len(df_res)}, Avg customer spend: INR {avg_val:.2f}")
        elif "Q6" in name:
            diff_rank_row = (df_res["rank_val"] != df_res["row_num"]).sum()
            print(f"Rows with RANK() != ROW_NUMBER(): {diff_rank_row}")
            top_per_city = df_res[df_res["rank_val"] == 1].sort_values("city")
            for _, r in top_per_city.iterrows():
                print(f"  {r['city']}: {r['customer_id']} ({r['total_spend']:.2f})")
        elif "Q7" in name:
            print(f"Daily count: {len(df_res)}, Final Cumulative Revenue: INR {df_res.iloc[-1]['cumulative_revenue']:.2f}")

    # 7. Pandas Cross-Check & EDA
    print("\n==================================================================")
    print("PANDAS CROSS-CHECK & EXPLORATORY DATA ANALYSIS")
    print("==================================================================")
    completed_df = orders[orders["status"] == "completed"].merge(cust[["customer_id", "city"]], on="customer_id", how="left")
    
    # Pivot table check
    pt = pd.pivot_table(completed_df, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
    grand_total = pt.loc["All", "All"]
    print(f"Pivot Table Grand Total: INR {grand_total:.2f}")

    # Groupby city aggregation
    city_agg = completed_df.groupby("city")["order_amount"].agg(sum="sum", mean="mean", count="count").round(2)
    print("\nCity Aggregation (sum, mean, count):\n", city_agg)

    # Order Amount Distribution Metrics
    amt_series = completed_df["order_amount"]
    mean_val = amt_series.mean()
    median_val = amt_series.median()
    skew_val = amt_series.skew()
    q1 = amt_series.quantile(0.25)
    q3 = amt_series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_below = (amt_series < lower_fence).sum()
    outliers_above = (amt_series > upper_fence).sum()

    print("\nOrder Amount Distribution:")
    print(f"  Mean: INR {mean_val:.2f}")
    print(f"  Median: INR {median_val:.2f}")
    print(f"  Skewness: {skew_val:.2f}")
    print(f"  Q1: INR {q1:.2f}, Q3: INR {q3:.2f}, IQR: INR {iqr:.2f}")
    print(f"  Lower Fence: INR {lower_fence:.2f} (Outliers below: {outliers_below})")
    print(f"  Upper Fence: INR {upper_fence:.2f} (Outliers above: {outliers_above})")

    # Correlations
    corr_vars = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_matrix = completed_df[corr_vars].corr()
    print("\nCorrelation Matrix:\n", corr_matrix.round(4))

    conn.close()
    print("\nPart A completed successfully.")

if __name__ == "__main__":
    run_part_a()
