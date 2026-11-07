# Instamart Customer Intelligence & Support Copilot

An end-to-end, reproducible data intelligence and customer support copilot designed for Swiggy Instamart's quick-commerce operations across six metro cities (Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, and Pune).

---

## Overview

As quick-commerce order volumes expand, weekly leadership reviews face three recurrent operational bottlenecks:
1. **Unreliable Data Exports:** Raw transactional exports contain duplicate records, pending order versions, impossible delivery times, and unparseable monetary values.
2. **Customer Retention Blindspots:** Retention teams lack visibility into which valuable customers are on the verge of lapsing.
3. **Overwhelmed Customer Support Desk:** Support workflows are inundated with thousands of unstructured free-text tickets mixing refunds, fraud reports, app crashes, and delivery delays without automated triage or privacy protections.

This repository implements the **Customer Intelligence & Support Copilot** across four modular, interconnected parts:
- **Part A (Data Foundation):** Audits and standardises raw data, resolves anomalies using documented cleaning rules, stores relational tables into a local SQLite database (`data/instamart.db`), and validates key business metrics using SQL window functions, CTEs, and pandas.
- **Part B (Customer Segments & Lapse Prediction):** Computes customer RFM features, clusters customers into actionable business segments via K-Means and PCA, frames a leakage-free lapse prediction task, and trains tuned classification models with class-balancing experiments.
- **Part C (Support Ticket Triage with Guardrails):** Establishes an automated triage system featuring input length guardrails, strict PII redaction (email, phone, order ID), transient LLM error retry resilience with exponential backoff and jitter, structured JSON/keyword validation, token budgeting, and priority routing informed by customer segments.
- **Part D (Policy RAG Assistant & Refund Agent):** Builds a grounded policy question-answering assistant using chunking and TF-IDF vector retrieval, and constructs a multi-step ReAct autonomous refund agent equipped with SQLite and retrieval tools, stop-safely guardrails (iteration limit, token budget, execution timeout), and state checkpointing for crash recovery.

---

## Reproduce

Follow these numbered shell commands to reproduce all results from scratch in a clean environment:

```bash
# 1. Clone the repository and enter the project directory
git clone https://github.com/<user>/instamart-copilot.git
cd instamart-copilot

# 2. Create and activate an isolated Python 3.11 virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# 3. Upgrade pip and install exact pinned dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Generate deterministic synthetic datasets and policy documents
python generate_data.py

# 5. Run Part A: Data cleaning, SQLite storage, SQL queries, and EDA
python src/part_a.py

# 6. Validate Part A numeric checkpoints
python src/validate_checkpoints.py

# 7. Run Part B: RFM segmentation, clustering, and lapse prediction
python src/part_b.py

# 8. Run Part C: Support ticket triage, PII redaction, and routing
python src/part_c.py

# 9. Run Part D: Policy RAG evaluation and ReAct refund agent
python src/part_d.py

# (Optional) Alternatively, run the entire end-to-end pipeline in one command:
# python run_all.py
```

---

## Part A — Data Foundation: Clean, Store, Query

### 1. Raw Data Audit

The raw exports (`customers.csv`, `products.csv`, `orders.csv`, and `tickets.csv`) were audited for structural shape, schema datatypes, and missing values:

| Table | Rows | Columns | Columns with Missing Values (Count / %) |
| :--- | :--- | :--- | :--- |
| `customers.csv` | 1,200 | 5 | `age` (61 / 5.08%), `gender` (30 / 2.50%) |
| `products.csv` | 40 | 4 | None (0 / 0.00%) |
| `orders.csv` | 13,084 | 9 | `order_amount` (53 / 0.41%), `delivery_minutes` (2,541 / 19.42%) |
| `tickets.csv` | 164 | 5 | None (0 / 0.00%) |

#### Specific Audit Findings Table
| Audit Metric | Raw Count | Reference Checkpoint | Status |
| :--- | :--- | :--- | :--- |
| Raw `orders.csv` rows | 13,084 | 13,084 | Exact match |
| Exact duplicate order rows | 119 | 119 | Exact match |
| Orders with status `pending` | 656 | 656 | Exact match |
| Distinct `order_id` values | 12,309 | 12,309 | Exact match |
| Non-numeric `order_amount` entries | 120 | 120 | Exact match |
| Negative `delivery_minutes` (< 0) | 106 | 106 | Exact match |
| Blank `delivery_minutes` | 2,541 | 2,541 | Exact match |
| Blank `age` in `customers.csv` | 61 | 61 | Exact match |
| `age` outside range [10, 100] | 6 | 6 | Exact match |
| Distinct raw `city` spellings | 24 | 24 | Exact match |
| Distinct non-blank raw `gender` spellings | 10 | 10 | Exact match |
| Blank `gender` entries | 30 | 30 | Exact match |

---

### 2. Customer Cleaning Rules

1. **City Standardisation:** Stripped leading/trailing whitespace and converted strings to title-case. Exactly 6 metro cities remain:
   - `Pune`: 227
   - `Delhi`: 210
   - `Bengaluru`: 197
   - `Chennai`: 191
   - `Hyderabad`: 191
   - `Mumbai`: 184
2. **Gender Normalisation:** Stripped whitespace and mapped variants (`male`, `Male`, `M`, `m`, `MALE` -> `male`; `female`, `Female`, `F`, `f`, `FEMALE` -> `female`; blank/missing -> `unknown`):
   - `female`: 610
   - `male`: 560
   - `unknown`: 30
3. **Age Validation & Imputation:** Coerced age to numeric. Values outside [10, 100] (6 planted records: 280, -4, 0, 150, 999, 200) were converted to NaN. All 67 missing ages (61 blank + 6 out-of-bounds) were imputed using the median age of the customer's city via `groupby('city')['age'].transform()`. Post-cleaning: 0 missing ages; minimum age is 18.0 and maximum age is 60.0.
4. **Signup Date Conversion:** Converted `signup_date` to ISO datetime string format (`YYYY-MM-DD`).

---

### 3. Order Cleaning Rules

The order cleaning pipeline was executed in this exact sequence:

1. **Order Timestamp Parsing:** Converted `order_ts` to datetime objects.
2. **Exact Duplicate Removal:** Removed 119 exact duplicate rows (`drop_duplicates()`), reducing row count from 13,084 to 12,965.
3. **Pending Record Resolution:** Orders recorded multiple times with different lifecycle statuses (earlier pending records and later completed/cancelled records) were resolved by sorting on `order_ts` and retaining only the final record per `order_id` (`keep='last'`). This eliminated 656 pending records, leaving exactly 12,309 unique orders (one per `order_id`).
4. **Order Amount Correction:** Converted `order_amount` to numeric, setting 120 unparseable entries (e.g. `'N/A'`, `'error'`) to NaN. Missing amounts were recalculated and filled using catalog price: $\text{order\_amount} = \text{round}(\text{price\_inr} \times \text{quantity} \times (1 - \text{discount\_pct} / 100), 2)$. Post-cleaning missing amounts: 0.
5. **Delivery Duration Imputation:** Converted `delivery_minutes` to numeric and set 106 planted negative values (`-10`) to NaN. For completed orders, missing delivery durations were imputed with the median delivery duration of the corresponding product category among completed orders. For cancelled orders, `delivery_minutes` was left as NaN because cancelled orders were never delivered and imputing delivery time would introduce synthetic bias.

#### Clean Orders Verification
- Clean row count: 12,309 rows (one per `order_id`)
- Status counts: 11,110 completed, 1,199 cancelled, 0 pending
- Missing `order_amount`: 0
- Missing `delivery_minutes`: 0 among completed orders, 1,199 among cancelled orders
- Completed revenue sum: **INR 2,998,668.80**
- Checkpoint validation script (`src/validate_checkpoints.py`) output: `ALL CHECKPOINTS PASSED`

---

### 4. SQL Analytical Query Results (`sql/queries.sql`)

All queries were executed against `data/instamart.db` via `sqlite3` and `pd.read_sql_query`:

| Query ID | Description | Result Summary | Reference Match |
| :--- | :--- | :--- | :--- |
| **Q1** | WHERE completed & order_amount > 500, GROUP BY customer HAVING sum > 8000 | **30 qualifying customers** | Exact match (30) |
| **Q2** | Three-table join: Revenue by city and product category | **30 rows**, Total revenue: **INR 2,998,668.80**; Top cell: **Bengaluru / Electronics (INR 233,909.45)** | Exact match |
| **Q3** | Scalar subquery: Products priced above catalogue average | **11 products** priced above average (INR 397.78) | Exact match (11) |
| **Q4** | LEFT JOIN: Customers with zero completed orders | **25 customers** | Exact match (25) |
| **Q5** | CTE: Customers with completed spend above average customer spend | **550 customers**; Average customer spend: **INR 2,552.06** | Exact match |
| **Q6** | Window functions: RANK, DENSE_RANK, ROW_NUMBER by city spend | Top customer per city:<br>• Bengaluru: `C0082` (INR 12,856.20)<br>• Chennai: `C0930` (INR 10,151.40)<br>• Delhi: `C0291` (INR 13,742.60)<br>• Hyderabad: `C0743` (INR 12,730.05)<br>• Mumbai: `C0693` (INR 14,766.55)<br>• Pune: `C0426` (INR 10,377.10)<br>**Exactly 8 rows** have `RANK() != ROW_NUMBER()`. | Exact match |
| **Q7** | Running total: Daily revenue with cumulative sum window | **183 daily rows** (2025-04-01 to 2025-09-30); Final cumulative revenue: **INR 2,998,668.80** | Exact match |

> **Reason for RANK() != ROW_NUMBER() in Q6:** When two or more customers within the same city achieve identical completed spend totals, `RANK()` assigns them the same ranking number and skips subsequent ranks, whereas `ROW_NUMBER()` arbitrarily breaks ties to assign strictly distinct, sequential integer ranks.

---

### 5. pandas Cross-Check and Exploratory Data Analysis

#### City Revenue Aggregation Cross-Check
Rebuilding Q2 via `pd.pivot_table(completed_orders, values='order_amount', index='city', columns='category', aggfunc='sum', margins=True)` yielded a grand total of **INR 2,998,668.80**, matching SQL exactly.

Named aggregation by city:
- **Bengaluru:** sum INR 523,643.40, mean INR 276.47, count 1,894
- **Chennai:** sum INR 463,629.15, mean INR 272.72, count 1,700
- **Delhi:** sum INR 502,829.80, mean INR 243.62, count 2,064
- **Hyderabad:** sum INR 528,589.90, mean INR 284.95, count 1,855
- **Mumbai:** sum INR 479,058.50, mean INR 297.00, count 1,613
- **Pune:** sum INR 500,918.05, mean INR 252.48, count 1,984

#### Completed Order Amount Distribution Metrics
- **Mean:** INR 269.91
- **Median:** INR 135.00
- **Skewness:** 4.61 (heavily right-skewed)
- **Q1:** INR 64.80 | **Q3:** INR 297.00 | **IQR:** INR 232.20
- **Lower Fence ($Q1 - 1.5\text{IQR}$):** INR -283.50 (Outliers below: 0)
- **Upper Fence ($Q3 + 1.5\text{IQR}$):** INR 645.30 (Outliers above: 897)

#### Correlation Matrix
| Variable | `order_amount` | `discount_pct` | `quantity` | `delivery_minutes` |
| :--- | :--- | :--- | :--- | :--- |
| `order_amount` | 1.0000 | **-0.0951** | +0.0873 | -0.0031 |
| `discount_pct` | -0.0951 | 1.0000 | -0.0004 | -0.0098 |
| `quantity` | +0.0873 | -0.0004 | 1.0000 | -0.0126 |
| `delivery_minutes` | -0.0031 | -0.0098 | -0.0126 | 1.0000 |

#### Analytical Synthesis (< 200 words)
This dataset captures 11,110 completed grocery and electronics orders across six Indian metros over six months. Order spending is heavily right-skewed (mean INR 269.91 vs median INR 135.00, skewness 4.61), driven by frequent small daily grocery purchases alongside occasional high-ticket electronics purchases. Exactly 897 orders exceed the upper fence (INR 645.30) up to INR 3,998.00; these represent legitimate premium transactions rather than data corruption. Linear correlation across variables is negligible: the largest correlation in absolute value is between order amount and discount percentage at −0.0951 (order amount with quantity is +0.0873). A correlation of this minor magnitude does not establish causation; higher basket values occasionally receive percentage discounts, but delivery time and basket quantity remain largely independent of order spend.

---

## Part B — Customer Segments and Lapse Prediction

### 1. RFM Metrics (Snapshot: 2025-10-01 00:00:00)
For all 1,175 customers with $\ge 1$ completed order:
- **Recency (days):** Median = 19.0, Min = 0, Max = 180
- **Frequency (order count):** Mean = 9.46, Max = 33
- **Monetary (completed spend):** Mean = INR 2,552.06, Max = INR 14,766.55

---

### 2. K-Means Clustering Sweep & Selection

| $K$ | Inertia | Silhouette Score | Business Interpretability |
| :--- | :--- | :--- | :--- |
| 2 | 1,820.8 | 0.465 | Coarse split into active vs inactive; misses spend differences |
| 3 | 1,268.2 | 0.445 | Separates activity but conflates high-ticket and frequent shoppers |
| **4** | **743.1** | **0.526** | **Optimal elbow & peak silhouette; isolates premium spenders cleanly** |
| 5 | 617.4 | 0.480 | Subdivides churned cohort into redundant sub-clusters |
| 6 | 503.2 | 0.482 | Diminishing structural return |
| 7 | 408.5 | 0.416 | Over-fragmented segments |
| 8 | 366.5 | 0.412 | Excessive fragmentation, low actionability |

> **Why inertia alone cannot choose $K$:** Inertia measures within-cluster sum of squared distances to centroids and monotonically decreases as $K$ increases because adding centroids always reduces distances (reaching 0 when $K=N$). Therefore, inertia alone cannot select $K$; the elbow inflection point, silhouette coefficient, and business utility must guide model selection.

---

### 3. Segment Profiles, Action Plan, and PCA Projection ($K=4$)

> *Note: Cluster numbers are arbitrary mathematical labels with no inherent ranking.*

| Segment ID | Business Segment Name | Size | Mean Recency | Mean Frequency | Mean Monetary | Is Premium? | Recommended Business Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | **Occasional Regulars** | 436 | 28.9 days | 3.8 | INR 1,057.62 | False | Basket-builder coupons and weekly category replenishment prompts |
| 1 | **High-Value Spenders** | 138 | 24.1 days | 6.8 | **INR 7,129.70** | **True** | Priority delivery, concierge VIP support desk, premium device promos |
| 2 | **Frequent Champions** | 419 | 8.8 days | 19.4 | INR 3,410.02 | False | Subscription loyalty programs (e.g. Instamart One) and dairy passes |
| 3 | **At-Risk / Lapsed** | 182 | 109.0 days | 2.2 | INR 685.99 | False | Win-back push notifications with aggressive re-activation discounts |

- **PCA Projection:** Applied `PCA(n_components=2)` to scaled RFM columns:
  - PC1 explained variance: **0.634**
  - PC2 explained variance: **0.218**
  - Total explained variance: **0.852**
- Saved to: `outputs/customer_segments.csv`

---

### 4. Lapse Prediction Cohort & Features (No Data Leakage)

- **Cutoff Date:** `2025-09-01 00:00:00`. Evaluation window: `2025-09-01` to `2025-09-30`.
- **Cohort:** 1,156 customers who placed $\ge 1$ completed order prior to the cutoff.
- **Label (`lapsed`):** 1 if customer placed zero completed orders in September 2025, else 0.
  - Cohort size: **1,156** | Lapsed count: **443 (38.32%)** | Active: 713 (61.68%)
- **Features Constructed Exclusively Before Cutoff:** `recency`, `frequency`, `monetary`, `avg_discount_pct`, `cancellation_rate`, `avg_delivery_minutes`, `favourite_category`, `city`, `age`, `gender`, `tenure_days`.
  - *No post-cutoff transactions and no Part B clustering labels were included in features.*
- **Data Split:** `train_test_split(test_size=0.2, stratify=y, random_state=42)`:
  - Training set: **924 rows** (354 lapsed, 570 retained)
  - Test set: **232 rows** (89 lapsed, 143 retained)
- **Majority Class Baseline (predict always 0):** Accuracy = **61.64%**, Lapsed F1 = **0.0000**.

---

### 5. Preprocessing & Leakage Prevention

- Numeric features (`recency`, `frequency`, `monetary`, `avg_discount_pct`, `cancellation_rate`, `avg_delivery_minutes`, `age`, `tenure_days`) were scaled using `StandardScaler`.
- Categorical features (`favourite_category`, `city`, `gender`) were encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- All encoders and scalers were enclosed inside scikit-learn `Pipeline` and `ColumnTransformer` objects fitted strictly on training data.
- **Verification of Training-Only Fitting:**
  - Learned frequency mean (train only): **8.1147**
  - Full-cohort frequency mean: **8.0285**
  > *These values differ because the training set is a random 80% stratified sample of the cohort, naturally displaying sample variance relative to the population mean.*
- Verified zero missing values across all feature columns.

---

### 6. Model Tuning and Test Set Evaluation

Both models were tuned using 5-fold cross-validation on the training set optimizing for positive class (lapsed) F1-score:
- **Logistic Regression:** Grid over $C \in \{0.01, 0.1, 1, 10, 100\}$. Best $C = 100$ (CV F1 = 0.6506).
- **Random Forest:** Grid over $\text{max\_depth} \in \{3, 5, 8\}$ and $\text{min\_samples\_leaf} \in \{5, 10, 20\}$ (45 fits). Best params: `max_depth=8`, `min_samples_leaf=10` (CV F1 = 0.6865).

#### Test Set Evaluation (Untouched Test Set, 232 Customers)
| Model | Lapsed Precision | Lapsed Recall | Lapsed F1 | Accuracy | Confusion Matrix [TN, FP / FN, TP] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Majority)** | 0.0000 | 0.0000 | 0.0000 | 61.64% | `[[143, 0], [89, 0]]` |
| **Tuned Logistic Regression** | 0.6444 | 0.6517 | 0.6480 | 72.84% | `[[111, 32], [31, 58]]` |
| **Tuned Random Forest** | 0.6404 | 0.6404 | **0.6404** | 72.41% | `[[111, 32], [32, 57]]` |

#### Model Selection Justification
We choose the **Tuned Random Forest** as the operational model (persisted to `outputs/lapse_model.joblib`). In customer retention, the business cost of a false negative (failing to detect a high-value customer about to churn) is substantially greater than the marginal marketing expense of a false positive (sending an automated re-engagement promotion to an already loyal customer). Random Forest captures non-linear interactions between recency, cancellation rate, and delivery delays without suffering from extreme coefficient swings, achieving balanced test precision and recall (64.04%) and outperforming the baseline by 64 percentage points in F1.

---

### 7. Class-Balance Experiment

To address class imbalance (354 lapsed vs 570 retained in train), we oversampled lapsed training rows with replacement (`random_state=42`) to reach exactly 570 rows per class (1,140 total):

| Setup | Training Rows | Lapsed Precision (Test) | Lapsed Recall (Test) | Lapsed F1 (Test) |
| :--- | :--- | :--- | :--- | :--- |
| Unbalanced Logistic Regression | 924 (354 lapsed) | 0.6444 | 0.6517 | 0.6480 |
| **Balanced Logistic Regression** | **1,140 (570 lapsed)** | **0.5891** | **0.8539** | **0.6972** |

> **Business Impact:** Recall surged dramatically from **0.6517 to 0.8539** (+20.2 percentage points) while precision experienced only a minor drop to 0.5891. For the Instamart retention team, this model configuration captures over 85% of at-risk customers, allowing timely intervention campaigns before habit abandonment sets in.

---

### 8. Model Interpretation, Persistence, and Illustrative Retention Priority

#### Random Forest Top 5 Feature Importances
1. `monetary` (0.2778): Total historical spend reflects customer investment and engagement tier.
2. `frequency` (0.2768): Purchase cadence is the primary operational anchor of habitual quick-commerce ordering.
3. `recency` (0.1112): Longer gap since the last pre-cutoff purchase signals weakening purchase momentum.
4. `avg_delivery_minutes` (0.0616): Chronic delivery friction directly drives churn.
5. `avg_discount_pct` (0.0532): Deal-seeking behaviour correlates with platform churn once incentives diminish.

#### Logistic Regression Coefficients
- `recency` (+0.0396): Positive sign indicates higher days since last order increases lapse odds.
- `frequency` (-1.6982): Strong negative sign shows higher order cadence strongly protects against churn.
- `monetary` (+0.2846): Positive sign reflecting that occasional bulky buyers lapse more easily than frequent daily purchasers.

#### Model Persistence
The final pipeline was saved via `joblib.dump(best_rf, "outputs/lapse_model.joblib")`. Reloading the model with `joblib.load` reproduced the exact predictions on the test set (`assert np.array_equal(y_pred_rf, reloaded_preds)`).

#### Illustrative Retention Priority List
Test-set customers identified as High-Value Spenders (Premium) were ranked by model-predicted lapse probability:

| Customer ID | Predicted Lapse Probability | Actually Lapsed |
| :--- | :--- | :--- |
| `C0345` | 0.5489 | 0 |
| `C0377` | 0.5362 | 0 |
| `C0034` | 0.5284 | 0 |
| `C0930` | 0.5276 | 0 |
| `C0094` | 0.5188 | 0 |
| `C1008` | 0.4907 | 1 |
| `C0376` | 0.4852 | 0 |
| `C0453` | 0.4578 | 0 |
| `C0305` | 0.4357 | 0 |
| `C0686` | 0.4287 | 0 |

Saved to: `outputs/retention_list.csv`. Exactly 1 customer in this top-10 list actually lapsed.
> *Notice:* The Premium flag derives from the whole-window RFM of Task 3 (which covers April–September 2025), so this list is an illustrative operational priority queue rather than a strictly leakage-free holdout test.

---

## Part C — Support Ticket Triage with Guardrails

### 1. Input Guardrails & PII Redaction
- **Empty / Whitespace Check:** 4 tickets had empty or whitespace-only bodies; assigned `skipped_empty` and never dispatched to the LLM.
- **Input Length Token Budget:** Evaluated using `mock_llm.est_tokens(text)`. Exactly 4 tickets exceeded `MAX_INPUT_TOKENS = 400` (each estimated at 734 tokens); assigned `flagged_long` and routed to human review without LLM invocation.
- **PII Redaction Regexes (applied in strict order):**
  1. Email: `r"[\w.+-]+@[\w-]+\.[\w.]+"` $\rightarrow$ `[EMAIL]` (55 substitutions)
  2. Phone: `r"\b[6-9]\d{9}\b"` $\rightarrow$ `[PHONE]` (91 substitutions)
  3. Order ID: `r"ORD-\d{6}"` $\rightarrow$ `[ORDER_ID]` (132 substitutions)
  - **Total Substitutions:** 278 across all 164 tickets.
  - Re-scanning confirmed **0 sensitive items remain**. Only redacted text was transmitted.

---

### 2. Resilient API Calls
Calls to `mock_llm.chat` simulate intermittent server unresponsiveness (`TransientLLMError: 503 server busy`) on every 9th invocation. A retry decorator with up to 3 retries, exponential backoff, and random jitter ($0.05 \times 2^n \times \text{uniform}(0.9, 1.15)$) caught and recovered all 19 transient errors across each 156-ticket batch. All retries succeeded on retry attempt 0 (waits between 0.045s and 0.0575s). Zero permanent failures occurred.

---

### 3. Rule-Based Output Validation
Model responses were parsed and verified in two stages:
1. **JSON Parser:** If valid JSON and `category` $\in \{\text{"billing"}, \text{"delivery"}, \text{"technical"}, \text{"fraud"}\}$, assigned status `classified_json`.
2. **Keyword Fallback:** If not valid JSON, extracted lowercase words via `re.findall(r"[a-z]+", text.lower())`. If exactly one valid category appeared (e.g. from chatty responses like `"The customer appears to have a billing-related problem."`), assigned status `classified_keyword`.
3. **Invalid Output Guard:** If invalid JSON with $\ne 1$ categories or valid JSON with an illegal category (e.g. `{"category": "shipping"}`), assigned status `flagged_invalid_output` and routed to human review.

Status counts for 156 model-evaluated tickets:
- `classified_json`: **135**
- `classified_keyword`: **12**
- `flagged_invalid_output`: **9**

---

### 4. Zero-Shot vs Few-Shot Performance Comparison

| Metric | Zero-Shot Pipeline | Few-Shot Pipeline |
| :--- | :--- | :--- |
| Eligible Tickets Evaluated | 156 | 156 |
| Tickets Successfully Classified | 147 (135 JSON + 12 keyword) | 147 (135 JSON + 12 keyword) |
| Flagged for Human Review | 9 invalid + 4 empty + 4 long = 17 | 9 invalid + 4 empty + 4 long = 17 |
| **Classification Accuracy** | **85.71% (126 / 147)** | **100.00% (147 / 147)** |
| Total Tokens Used | 11,740 tokens | 18,916 tokens |
| Tokens per Classified Ticket | **79.9 tokens** | **128.7 tokens** |

#### Zero-Shot Confusion Table
| True Category \ Predicted | `billing` | `delivery` | `fraud` | `technical` |
| :--- | :--- | :--- | :--- | :--- |
| `billing` | **38** | 0 | 0 | 0 |
| `delivery` | 0 | **36** | 0 | 0 |
| `fraud` | **11** | 0 | **23** | 0 |
| `technical` | **10** | 0 | 0 | **29** |

#### Justification for Few-Shot Prompting
In our benchmark, the zero-shot prompt mistakenly categorised 11 fraud tickets and 10 technical tickets as billing because vocabulary like *"charged"* or *"deducted"* triggered greedy keyword matches. Few-shot prompting achieved 100% classification accuracy on all 147 classified tickets at an incremental cost of 48.8 tokens per ticket (128.7 vs 79.9). In high-risk workflows where misclassifying unauthorized card transactions compromises security compliance, the ~61% extra token expenditure is thoroughly justified by eliminating fraud misrouting.

---

### 5. Token Budgeting and Final Ticket Routing

#### Token Budget Demonstration
Running the triage pipeline with `TOKEN_BUDGET = 6000` halted LLM calls once accumulated consumption reached the limit, safely setting **76 tickets** to status `budget_exhausted` and routing them to human review. Saved to: `outputs/triage_budget_demo.csv`.

#### Final Ticket Routing (`outputs/triage_results.csv`)
Using the few-shot classifier with `TOKEN_BUDGET = 100000`, all 164 tickets were routed using customer segment context from Part B:
- **`human_review` (Priority: NORMAL):** 17 tickets (4 empty, 4 long, 9 invalid LLM outputs).
- **`security_team` (Priority: HIGH):** 34 tickets (all confirmed fraud cases).
- **`premium_desk` (Priority: HIGH):** 11 tickets (non-fraud requests from High-Value Spenders).
- **`support_queue` (Priority: NORMAL):** 102 tickets (general billing, delivery, technical).

---

## Part D — Policy RAG Assistant and Refund Agent

### 1. Policy Document Chunking & TF-IDF Retrieval
The five internal policy text files were chunked using a 40-word sliding window with a 10-word overlap (step = 30 words), stopping once a chunk reached or passed document end:
- Total chunks created: **15 chunks**
  - `refund_policy.txt`: 4 chunks
  - `delivery_policy.txt`: 3 chunks
  - `escalation_policy.txt`: 3 chunks
  - `payment_policy.txt`: 3 chunks
  - `privacy_policy.txt`: 2 chunks

Querying `retrieve("refund policy late delivery", k=3)` returned three chunks all originating from `refund_policy.txt` (similarities: 0.2719, 0.2514, 0.2454).

> **TF and IDF Definition & Limitation:** Term Frequency (TF) measures how frequently a term appears within a specific chunk, while Inverse Document Frequency (IDF) penalises terms that appear across many chunks to surface rare, informative words. A key limitation of TF-IDF is its reliance on strict lexical matching; it fails to capture semantic synonyms (e.g. *"money back"* vs *"refund"*) or morphological variations not handled by token stemming.

---

### 2. Grounding Evaluation: Overlap vs Zero-Overlap

We evaluated retrieval and answer fidelity across 14 benchmark questions (12 policy-covered, 2 out-of-scope) in `data/raw/policy_eval.csv`:

| Evaluation Setup | Chunk Count | Top-3 Doc Match (of 12) | Fact in Answer (of 12) | Unanswerable Fallback (of 2) |
| :--- | :--- | :--- | :--- | :--- |
| **Overlap = 10 words** | **15** | **12 / 12 (100%)** | **12 / 12 (100%)** | **2 / 2 (100%)** |
| **Overlap = 0 words** | **13** | 12 / 12 (100%) | **10 / 12 (83.3%)** | 2 / 2 (100%) |

#### Analysis of Zero-Overlap Failures
Without overlap, 2 out of 12 answerable questions failed because critical factual clauses were cleaved directly across chunk boundaries:
1. *"What delivery time makes a completed order eligible for a full refund?"*
   - Expected: `"more than 45 minutes"`.
   - Result: Failed because *"If a completed order is delivered in more than 45 minutes, the customer"* ended chunk 0, while *"receives a full refund of the order amount."* began chunk 1.
2. *"How soon is money returned when a payment fails but is deducted?"*
   - Expected: `"48 hours"`.
   - Result: The timeframe was severed across chunk splits, producing a truncated response.

---

### 3. ReAct Refund Agent Evaluation (7 Scenarios)

The refund agent operates an autonomous ReAct loop equipped with five tools:
1. `get_order(order_id)`: Queries `data/instamart.db` for status, amount, and delivery time.
2. `search_policy(query)`: Performs TF-IDF retrieval over policy chunks.
3. `calculate_refund(order_amount, delivery_minutes)`: Calculates 100% refund for completed orders delivered in $> 45$ minutes; 0 otherwise.
4. `issue_refund(order_id, refund_inr)`: Issues automatic refund ($\le \text{INR } 1000$) and records to `outputs/refund_ledger.jsonl`.
5. `escalate_to_human(order_id, refund_inr)`: Escalates high-value refund ($> \text{INR } 1000$) to a human lead and records to `outputs/refund_ledger.jsonl`.

#### Seven Scenario Execution Results
| Scenario | Order / Case | Expected Outcome | Final Agent Status | Refund Amount | LLM Calls | Tokens Used | Match? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | Late delivery $\le 1000$ | `REFUND` | `REFUND` | **INR 128.25** | 5 | 957 | Exact |
| **S2** | Late delivery $\le 1000$ | `REFUND` | `REFUND` | **INR 135.00** | 5 | 957 | Exact |
| **S3** | Late delivery $> 1000$ | `ESCALATE` | `ESCALATE` | **INR 1,999.00** | 5 | 944 | Exact |
| **S4** | On-time delivery ($\le 45\text{m}$) | `DENIED` | `DENIED` | INR 0.00 | 4 | 653 | Exact |
| **S5** | Cancelled order | `DENIED` | `DENIED` | INR 0.00 | 3 | 373 | Exact |
| **S6** | Unknown order `ORD-999999` | `NOT_FOUND` | `NOT_FOUND` | INR 0.00 | 2 | 96 | Exact |
| **S7** | Simulated loop `[LOOP]` | `FALLBACK` | `FALLBACK_MAX_ITERATIONS` | INR 0.00 | 6 | 3,072 | Exact |

All executions logged to `outputs/agent_runs.jsonl`. Approved refunds and escalations written to `outputs/refund_ledger.jsonl`.

---

### 4. Stop-Safely Demonstrations

1. **Token Limit Guardrail Demo:** Executed scenario S1 with `token_limit = 200`. The agent stopped immediately after Call 1 before executing tools, returning status: **`FALLBACK_TOKEN_LIMIT`**.
2. **Timeout Guardrail Demo:** Executed scenario S1 with `time_limit_seconds = 0`. The agent stopped prior to making any network/model calls, returning status: **`FALLBACK_TIMEOUT`**.

---

### 5. Resumable Execution & Crash Recovery Demonstration

To demonstrate fault-tolerant state recovery, scenario S1 was launched with a simulated runtime crash triggered immediately after tool execution #2 (`search_policy`), following disk checkpoint persistence to `state/s1_resumable_demo.json`. Calling `resume_agent("s1_resumable_demo")` successfully restored messages, token counts, and previous tool executions, completing the workflow without repeating any tool:
- **Tools executed across crash and resume (exactly 4 in order):**
  1. `get_order` (pre-crash)
  2. `search_policy` (pre-crash, checkpoint saved)
  3. `calculate_refund` (post-resume)
  4. `issue_refund` (post-resume)
- **Final Resumed Status:** `REFUND` (INR 128.25 issued).

---

### 6. Architectural Trade-off: ReAct vs Plan-and-Execute (< 250 words)

In this project, our ReAct agent dynamic loop required 5 LLM calls for full refund approvals (S1/S2/S3), 4 calls for on-time denials (S4), 3 for cancelled orders (S5), and 2 for non-existent orders (S6). Total tokens scaled linearly with interaction depth (96 tokens for S6 up to 957 tokens for S1). 

A Plan-and-Execute architecture generates the entire sequence of tool invocations upfront in a single planning prompt and executes them sequentially, replanning only when encountering unexpected observations. For standard happy-path scenarios, Plan-and-Execute saves tokens and reduces latency by consolidating multi-turn reasoning into 1–2 LLM calls. However, when handling real-world quick-commerce anomalies—such as an invalid order ID (`NOT_FOUND`), a cancelled order ineligible for refund, or a missing customer identifier—Plan-and-Execute generates unnecessary tool steps that must be aborted and replanned, incurring replanning overhead. 

For Instamart’s automated refund workflow, the **ReAct design is distinctly superior**. Financial operations demand strict precondition gating: verifying that an order exists and was completed before retrieving policies or calculating payouts prevents accidental disbursements. ReAct’s step-by-step observation verification guarantees that execution halts immediately upon encountering missing orders (2 calls) or cancelled statuses (3 calls), ensuring complete financial safety and auditable tool execution.

---

## Limitations and Next Steps

1. **Synthetic Data Simplifications:** The synthetic dataset features stationary customer behaviour and Poisson order distributions. Real-world customer ordering habits exhibit strong weekly seasonality, pay-day surges, and hyper-local rain weather shocks that require time-decayed feature weighting.
2. **Illustrative Whole-Window Segment Labeling:** As documented in Part B Task 8, the `is_premium` flag in `outputs/retention_list.csv` was derived from the full 6-month RFM table (which encompasses the September lapse window). While the lapse prediction model itself was trained strictly on pre-cutoff data without leakage, the retention prioritization list is illustrative of an operational dashboard rather than a leak-free backtest. Future iterations should compute dynamic, rolling 90-day RFM snapshots strictly prior to the cutoff date.
3. **Lexical Retrieval Limitations:** The TF-IDF vectorizer operates on exact lexical n-grams. In production, semantic dense embeddings (e.g. sentence transformers) coupled with approximate nearest neighbour search (FAISS) would capture colloquial phrasing (e.g. *"food went bad"*, *"rider was rude"*).
4. **Human-in-the-Loop Governance:** High-value refunds (> INR 1,000) and low-confidence classifications are currently flagged to `human_review` and `escalate_to_human`. Future engineering should integrate a production dashboard allowing agents to approve pending ledger items and feed corrected labels back into model re-training loops.
