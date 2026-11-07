"""src/part_b.py - Customer Segments and Lapse Prediction
Clusters customers using RFM analysis, trains lapse prediction models without data leakage,
and produces customer segments, trained model, and an illustrative retention priority list.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run_part_b():
    os.makedirs("outputs", exist_ok=True)
    clean_dir = os.path.join("data", "clean")
    
    # 1. Load Clean Data
    cust = pd.read_csv(os.path.join(clean_dir, "customers.csv"))
    prod = pd.read_csv(os.path.join(clean_dir, "products.csv"))
    orders = pd.read_csv(os.path.join(clean_dir, "orders.csv"))
    
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    
    # Merge category for favourite category feature
    orders = orders.merge(prod[["product_id", "category"]], on="product_id", how="left")
    
    completed_orders = orders[orders["status"] == "completed"].copy()

    print("==================================================================")
    print("PART B: CUSTOMER SEGMENTS AND LAPSE PREDICTION")
    print("==================================================================")

    # Task 1: Build RFM Table as of snapshot 2025-10-01 00:00:00
    snapshot = pd.Timestamp("2025-10-01 00:00:00")
    rfm = completed_orders.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)

    print("\n--- Task 1: RFM Checkpoints ---")
    print(f"Customer count with >=1 completed order: {len(rfm)}")
    print(f"Recency: median={rfm['recency'].median()}, min={rfm['recency'].min()}, max={rfm['recency'].max()}")
    print(f"Frequency: mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()}")
    print(f"Monetary: mean=INR {rfm['monetary'].mean():.2f}, max=INR {rfm['monetary'].max():.2f}")

    # Task 2: Standardise and Cluster with K-Means (K=2 to 8)
    scaler_rfm = StandardScaler()
    X_rfm = scaler_rfm.fit_transform(rfm[["recency", "frequency", "monetary"]])

    print("\n--- Task 2: K-Means Clustering Sweep ---")
    kmeans_results = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_rfm)
        inertia = km.inertia_
        sil = silhouette_score(X_rfm, labels)
        kmeans_results.append((k, inertia, sil))
        print(f"  K={k}: Inertia={inertia:7.1f}, Silhouette Score={sil:.3f}")

    # Task 3: Profile, Name and Project Segments (K=4)
    km4 = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm["segment_id"] = km4.fit_predict(X_rfm)

    seg_summary = rfm.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_freq=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    # Identify Premium segment (highest mean monetary)
    prem_seg_id = seg_summary.loc[seg_summary["mean_monetary"].idxmax(), "segment_id"]

    def assign_segment_name(row):
        sid = row["segment_id"]
        if sid == prem_seg_id:
            return "High-Value Spenders"
        elif row["mean_freq"] > 15:
            return "Frequent Champions"
        elif row["mean_recency"] > 40:
            return "At-Risk / Lapsed"
        else:
            return "Occasional Regulars"

    segment_name_map = {}
    for _, row in seg_summary.iterrows():
        segment_name_map[row["segment_id"]] = assign_segment_name(row)

    rfm["segment_name"] = rfm["segment_id"].map(segment_name_map)
    rfm["is_premium"] = rfm["segment_id"] == prem_seg_id

    print("\n--- Task 3: Segment Profiles (K=4) ---")
    for _, row in seg_summary.iterrows():
        sid = row["segment_id"]
        name = segment_name_map[sid]
        is_prem = (sid == prem_seg_id)
        print(f"  Segment {sid} - {name:20}: size={int(row['size'])}, recency={row['mean_recency']:.1f}d, freq={row['mean_freq']:.1f}, monetary=INR {row['mean_monetary']:.2f}, is_premium={is_prem}")

    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_rfm)
    var_ratios = pca.explained_variance_ratio_
    print(f"\nPCA Explained Variance Ratios: PC1={var_ratios[0]:.3f}, PC2={var_ratios[1]:.3f}, Total={var_ratios.sum():.3f}")

    # Save customer segments
    out_segments_path = os.path.join("outputs", "customer_segments.csv")
    rfm_output = rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
    rfm_output.to_csv(out_segments_path, index=False)
    print(f"Wrote customer segments to {out_segments_path}")

    # Task 4: Define the Lapse Problem
    cutoff = pd.Timestamp("2025-09-01 00:00:00")
    end_eval_window = pd.Timestamp("2025-09-30 23:59:59")

    # Cohort: customers with >= 1 completed order before cutoff
    pre_orders = orders[orders["order_ts"] < cutoff].copy()
    pre_completed = pre_orders[pre_orders["status"] == "completed"].copy()
    cohort_cids = sorted(pre_completed["customer_id"].unique())
    print(f"\n--- Task 4: Lapse Cohort & Labels ---")
    print(f"Cohort size (>= 1 completed order before 2025-09-01): {len(cohort_cids)}")

    # Customers with >= 1 completed order between 2025-09-01 and 2025-09-30
    post_completed = orders[
        (orders["order_ts"] >= cutoff) &
        (orders["order_ts"] <= end_eval_window) &
        (orders["status"] == "completed")
    ]
    post_retained_cids = set(post_completed["customer_id"].unique())

    # Lapsed label = 1 if no completed order in September 2025, else 0
    y_lapsed = pd.Series([1 if cid not in post_retained_cids else 0 for cid in cohort_cids], index=cohort_cids)
    print(f"Lapsed count: {y_lapsed.sum()} ({y_lapsed.mean()*100:.2f}%)")

    # Feature Engineering strictly before cutoff
    rec_series = pre_completed.groupby("customer_id")["order_ts"].max()
    freq_series = pre_completed.groupby("customer_id")["order_id"].nunique()
    mon_series = pre_completed.groupby("customer_id")["order_amount"].sum()
    disc_series = pre_completed.groupby("customer_id")["discount_pct"].mean()
    deliv_series = pre_completed.groupby("customer_id")["delivery_minutes"].mean()

    # Cancellation rate
    all_pre_counts = pre_orders.groupby("customer_id")["order_id"].nunique()
    canc_pre_counts = pre_orders[pre_orders["status"] == "cancelled"].groupby("customer_id")["order_id"].nunique()
    canc_rate_series = (canc_pre_counts / all_pre_counts).fillna(0.0)

    # Favourite category (mode)
    fav_cat_series = pre_completed.groupby("customer_id")["category"].agg(lambda s: s.mode()[0])

    cust_map = cust.set_index("customer_id")

    feature_rows = []
    for cid in cohort_cids:
        c_row = cust_map.loc[cid]
        feature_rows.append({
            "customer_id": cid,
            "recency": (cutoff - rec_series[cid]).days,
            "frequency": freq_series[cid],
            "monetary": mon_series[cid],
            "avg_discount_pct": disc_series[cid],
            "cancellation_rate": canc_rate_series.get(cid, 0.0),
            "avg_delivery_minutes": deliv_series[cid],
            "favourite_category": fav_cat_series[cid],
            "city": c_row["city"],
            "age": c_row["age"],
            "gender": c_row["gender"],
            "tenure_days": (cutoff - c_row["signup_date"]).days,
            "lapsed": y_lapsed[cid]
        })
    df_cohort = pd.DataFrame(feature_rows)
    print(f"Cohort features table shape: {df_cohort.shape}")
    assert df_cohort.isna().sum().sum() == 0, "Missing values found in cohort features."

    feature_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate",
        "avg_delivery_minutes", "favourite_category", "city", "age", "gender", "tenure_days"
    ]
    X = df_cohort[feature_cols]
    y = df_cohort["lapsed"]

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_cohort.index, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # Baseline: Always predict 0 (not lapsed)
    y_base = np.zeros(len(y_test))
    base_acc = accuracy_score(y_test, y_base)
    base_f1 = f1_score(y_test, y_base, zero_division=0)
    print(f"Majority Baseline - Accuracy: {base_acc*100:.2f}%, Lapsed F1: {base_f1:.4f}")

    # Task 5: Preprocessing without Leakage
    num_cols = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
    cat_cols = ["favourite_category", "city", "gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
        ]
    )

    # Verification of training-only fit
    scaler_demo = StandardScaler()
    scaler_demo.fit(X_train[["frequency"]])
    print(f"\nFrequency Scaler Mean (Train only): {scaler_demo.mean_[0]:.4f} vs Full Cohort Mean: {df_cohort['frequency'].mean():.4f}")

    # Task 6: Train, Tune and Evaluate
    print("\n--- Task 6: Hyperparameter Tuning ---")
    # Logistic Regression
    pipe_lr = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    grid_lr = GridSearchCV(pipe_lr, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
    grid_lr.fit(X_train, y_train)
    print(f"Logistic Regression: Best Params={grid_lr.best_params_}, 5-Fold CV F1={grid_lr.best_score_:.4f}")

    # Random Forest
    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    param_rf = {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    grid_rf = GridSearchCV(pipe_rf, param_rf, cv=5, scoring="f1")
    grid_rf.fit(X_train, y_train)
    print(f"Random Forest: Best Params={grid_rf.best_params_}, 5-Fold CV F1={grid_rf.best_score_:.4f}")

    # Evaluate on Untouched Test Set
    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    print("\nTest Evaluation - Logistic Regression:")
    print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print(f"  Precision: {precision_score(y_test, y_pred_lr):.4f}, Recall: {recall_score(y_test, y_pred_lr):.4f}, F1: {f1_score(y_test, y_pred_lr):.4f}")

    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    print("\nTest Evaluation - Random Forest:")
    print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}, Recall: {recall_score(y_test, y_pred_rf):.4f}, F1: {f1_score(y_test, y_pred_rf):.4f}")

    # Task 7: Class-Balance Experiment
    print("\n--- Task 7: Class-Balance Experiment ---")
    train_combined = pd.concat([X_train, y_train], axis=1)
    train_class_0 = train_combined[train_combined["lapsed"] == 0]
    train_class_1 = train_combined[train_combined["lapsed"] == 1]
    print(f"Original Train counts: Class 0={len(train_class_0)}, Class 1={len(train_class_1)}")

    n_oversample = len(train_class_0) - len(train_class_1)
    oversampled_1 = train_class_1.sample(n=n_oversample, replace=True, random_state=42)
    balanced_train = pd.concat([train_combined, oversampled_1], axis=0)
    print(f"Balanced Train counts: Class 0={len(balanced_train[balanced_train['lapsed']==0])}, Class 1={len(balanced_train[balanced_train['lapsed']==1])}")

    X_train_bal = balanced_train[feature_cols]
    y_train_bal = balanced_train["lapsed"]

    pipe_lr_bal = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=grid_lr.best_params_["clf__C"], max_iter=1000, random_state=42))
    ])
    pipe_lr_bal.fit(X_train_bal, y_train_bal)
    y_pred_bal = pipe_lr_bal.predict(X_test)

    print("\nTest Evaluation - Balanced Logistic Regression:")
    print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bal))
    print(f"  Precision: {precision_score(y_test, y_pred_bal):.4f}, Recall: {recall_score(y_test, y_pred_bal):.4f}, F1: {f1_score(y_test, y_pred_bal):.4f}")

    # Task 8: Interpret, Persist and Illustrative Retention Priority
    print("\n--- Task 8: Interpretability & Persistence ---")
    # Feature names after one-hot encoding
    cat_encoder = best_rf.named_steps["prep"].named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_cat_names)

    rf_clf = best_rf.named_steps["clf"]
    importances = pd.Series(rf_clf.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    print("Top 5 Feature Importances (Random Forest):")
    for feat, imp in importances.head(5).items():
        print(f"  {feat:25}: {imp:.4f}")

    lr_clf = best_lr.named_steps["clf"]
    coefs = pd.Series(lr_clf.coef_[0], index=all_feature_names)
    print("\nKey Feature Coefficients (Logistic Regression):")
    for feat in ["recency", "frequency", "monetary"]:
        if feat in coefs:
            print(f"  {feat:20}: coef={coefs[feat]:+.4f} (Sign: {'+' if coefs[feat] > 0 else '-'})")

    # Persist model
    model_path = os.path.join("outputs", "lapse_model.joblib")
    joblib.dump(best_rf, model_path)
    print(f"\nPersisted lapse model to {model_path}")

    # Reload model and verify identical predictions
    reloaded_model = joblib.load(model_path)
    reloaded_preds = reloaded_model.predict(X_test)
    assert np.array_equal(y_pred_rf, reloaded_preds), "Reloaded model failed to reproduce exact test predictions."
    print("Verification: Reloaded model reproduces test predictions exactly.")

    # Illustrative Retention List (Test set customers in Premium segment)
    test_cids = df_cohort.loc[test_idx, "customer_id"].values
    test_probs = best_rf.predict_proba(X_test)[:, 1]
    
    test_df_results = pd.DataFrame({
        "customer_id": test_cids,
        "lapse_probability": test_probs,
        "actually_lapsed": y_test.values
    })

    # Merge whole-window segment to identify Premium customers
    prem_cids = set(rfm[rfm["is_premium"]]["customer_id"])
    prem_test = test_df_results[test_df_results["customer_id"].isin(prem_cids)].copy()
    retention_list = prem_test.sort_values("lapse_probability", ascending=False).head(10)
    
    out_retention_path = os.path.join("outputs", "retention_list.csv")
    retention_list.to_csv(out_retention_path, index=False)
    print(f"\nWrote illustrative retention priority list ({len(retention_list)} customers) to {out_retention_path}")
    print(f"Actually lapsed count in top-10 list: {retention_list['actually_lapsed'].sum()} / {len(retention_list)}")

    print("\nPart B completed successfully.")

if __name__ == "__main__":
    run_part_b()
