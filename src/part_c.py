"""src/part_c.py - Support Ticket Triage with Guardrails
Applies privacy redaction, input guardrails, resilient LLM retries with backoff and jitter,
rule-based output validation, token budgeting, and segment-informed ticket routing.
"""

import os
import sys
import re
import json
import random
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
import mock_llm

CATS = ["billing", "delivery", "technical", "fraud"]

# Regular expressions for PII redaction
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_PHONE = re.compile(r"\b[6-9]\d{9}\b")
RE_ORDER = re.compile(r"ORD-\d{6}")

def redact_pii(text):
    """Applies email, phone, and order id redactions in exact order."""
    t, n_email = RE_EMAIL.subn("[EMAIL]", text)
    t, n_phone = RE_PHONE.subn("[PHONE]", t)
    t, n_order = RE_ORDER.subn("[ORDER_ID]", t)
    return t, n_email, n_phone, n_order

# System Prompts
PROMPT_ZERO = """You are a customer support ticket CLASSIFY agent.
Categorize the user ticket into exactly one of these categories: billing, delivery, technical, fraud.
Reply ONLY with a JSON object of the form:
{"category": "<one_of_the_four>", "reason": "<short explanation>"}"""

PROMPT_FEW = """You are a customer support ticket CLASSIFY agent.
Categorize the user ticket into exactly one of these categories: billing, delivery, technical, fraud.
Reply ONLY with a JSON object of the form:
{"category": "<one_of_the_four>", "reason": "<short explanation>"}

EXAMPLES:
Ticket: Unauthorized transaction on my account. Someone stole my card details.
{"category": "fraud", "reason": "unauthorized card transaction"}
Ticket: Where is my order? The rider has not arrived yet.
{"category": "delivery", "reason": "order delivery delay"}"""

def call_llm_with_retry(messages, ticket_id):
    """Retries up to 3 times on TransientLLMError with exponential backoff and jitter."""
    BASE_DELAY = 0.05
    for attempt in range(3):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError as err:
            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            print(f"[RETRY] Ticket {ticket_id} failed with {err}. Retry {attempt} after {wait:.4f}s")
    raise RuntimeError(f"Ticket {ticket_id} failed permanently after 3 retries.")

def parse_and_validate(raw_content):
    """Validates LLM response as JSON category or keyword fallback; flags invalid output."""
    # 1. Attempt JSON parsing
    try:
        data = json.loads(raw_content)
        if isinstance(data, dict):
            cat = str(data.get("category", "")).lower().strip()
            if cat in CATS:
                return "classified_json", cat
    except Exception:
        pass

    # 2. Keyword fallback extraction
    words = set(re.findall(r"[a-z]+", raw_content.lower()))
    found = [c for c in CATS if c in words]
    if len(found) == 1:
        return "classified_keyword", found[0]

    return "flagged_invalid_output", None

def run_triage_pipeline(tickets_df, system_prompt, token_budget=None):
    """Runs end-to-end triage over support tickets."""
    results = []
    running_tokens = 0
    MAX_INPUT_TOKENS = 400

    for _, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        txt = str(row["text"]) if pd.notna(row["text"]) else ""

        # Guardrail 1: Empty or whitespace text
        if not txt or not txt.strip():
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                "category": "", "tokens_used": 0
            })
            continue

        # Guardrail 2: Input length budget
        if mock_llm.est_tokens(txt) > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                "category": "", "tokens_used": 0
            })
            continue

        # Guardrail 3: Token budget check before call
        if token_budget is not None and running_tokens >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "tokens_used": 0
            })
            continue

        # Redact PII
        redacted_text, _, _, _ = redact_pii(txt)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text}
        ]

        resp = call_llm_with_retry(messages, tid)
        used = resp.usage.total_tokens
        running_tokens += used

        status, cat = parse_and_validate(resp.choices[0].message.content)
        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": cat if cat else "",
            "tokens_used": used
        })

    return pd.DataFrame(results), running_tokens

def run_part_c():
    os.makedirs("outputs", exist_ok=True)
    raw_tickets_path = os.path.join("data", "raw", "tickets.csv")
    df_tix = pd.read_csv(raw_tickets_path, keep_default_na=False)

    print("==================================================================")
    print("PART C: SUPPORT TICKET TRIAGE WITH GUARDRAILS")
    print("==================================================================")
    print(f"Loaded {len(df_tix)} tickets from {raw_tickets_path}")

    # Task 1 & 2: Audit Guardrails & PII Redaction
    tot_email, tot_phone, tot_order = 0, 0, 0
    empty_count, long_count = 0, 0
    MAX_INPUT_TOKENS = 400

    for txt in df_tix["text"]:
        if not txt or not txt.strip():
            empty_count += 1
        elif mock_llm.est_tokens(txt) > MAX_INPUT_TOKENS:
            long_count += 1
        
        _, e, p, o = redact_pii(txt)
        tot_email += e
        tot_phone += p
        tot_order += o

    print("\n--- Tasks 1 & 2: Input Guardrails & PII Redaction ---")
    print(f"Empty tickets: {empty_count}")
    print(f"Long tickets (>400 tokens): {long_count}")
    print(f"Redaction counts: Email={tot_email}, Phone={tot_phone}, Order ID={tot_order}, Total={tot_email + tot_phone + tot_order}")

    # Assertion: 0 sensitive items remain
    for txt in df_tix["text"]:
        red, _, _, _ = redact_pii(txt)
        assert not RE_EMAIL.search(red), f"Unredacted email found: {red}"
        assert not RE_PHONE.search(red), f"Unredacted phone found: {red}"
        assert not RE_ORDER.search(red), f"Unredacted order id found: {red}"
    print("Assertion Verified: 0 sensitive PII items remain across all redacted ticket texts.")

    # Task 6: Compare Zero-Shot and Few-Shot (No budget limit)
    print("\n--- Task 6: Zero-Shot vs Few-Shot Comparison ---")
    # Zero-Shot Run
    mock_llm.reset()
    print("Executing Zero-Shot run...")
    df_zero, tokens_zero = run_triage_pipeline(df_tix, PROMPT_ZERO)
    df_zero = df_zero.merge(df_tix[["ticket_id", "true_category"]], on="ticket_id")
    
    zero_status_counts = df_zero["status"].value_counts().to_dict()
    df_zero_class = df_zero[df_zero["status"].isin(["classified_json", "classified_keyword"])]
    acc_zero = (df_zero_class["category"] == df_zero_class["true_category"]).mean()

    print("\nZero-Shot Results:")
    print("  Status breakdown:", zero_status_counts)
    print(f"  Classified tickets: {len(df_zero_class)} / 156 eligible")
    print(f"  Accuracy on classified: {acc_zero*100:.2f}% ({int((df_zero_class['category'] == df_zero_class['true_category']).sum())} / {len(df_zero_class)})")
    print(f"  Total tokens: {tokens_zero}, Tokens per classified ticket: {tokens_zero / len(df_zero_class):.1f}")
    print("  Zero-Shot Confusion Table (True vs Predicted):\n", pd.crosstab(df_zero_class["true_category"], df_zero_class["category"]))

    # Few-Shot Run
    mock_llm.reset()
    print("\nExecuting Few-Shot run...")
    df_few, tokens_few = run_triage_pipeline(df_tix, PROMPT_FEW)
    df_few = df_few.merge(df_tix[["ticket_id", "true_category"]], on="ticket_id")
    
    few_status_counts = df_few["status"].value_counts().to_dict()
    df_few_class = df_few[df_few["status"].isin(["classified_json", "classified_keyword"])]
    acc_few = (df_few_class["category"] == df_few_class["true_category"]).mean()

    print("\nFew-Shot Results:")
    print("  Status breakdown:", few_status_counts)
    print(f"  Classified tickets: {len(df_few_class)} / 156 eligible")
    print(f"  Accuracy on classified: {acc_few*100:.2f}% ({int((df_few_class['category'] == df_few_class['true_category']).sum())} / {len(df_few_class)})")
    print(f"  Total tokens: {tokens_few}, Tokens per classified ticket: {tokens_few / len(df_few_class):.1f}")

    # Task 7(a): Token Budget Demonstration
    print("\n--- Task 7(a): Token Budget Demonstration ---")
    mock_llm.reset()
    DEMO_BUDGET = 6000
    df_budget_demo, _ = run_triage_pipeline(df_tix, PROMPT_ZERO, token_budget=DEMO_BUDGET)
    budget_exhausted_count = (df_budget_demo["status"] == "budget_exhausted").sum()
    print(f"Budget Demo (TOKEN_BUDGET={DEMO_BUDGET}): {budget_exhausted_count} tickets marked budget_exhausted.")
    
    out_demo_path = os.path.join("outputs", "triage_budget_demo.csv")
    df_budget_demo.to_csv(out_demo_path, index=False)
    print(f"Wrote budget demo results to {out_demo_path}")

    # Task 7(b): Final Routing with Segment Integration
    print("\n--- Task 7(b): Final Production Routing ---")
    segments_path = os.path.join("outputs", "customer_segments.csv")
    seg_df = pd.read_csv(segments_path)
    seg_map = seg_df.set_index("customer_id")

    mock_llm.reset()
    df_final, _ = run_triage_pipeline(df_tix, PROMPT_FEW, token_budget=100000)

    routed_records = []
    for _, row in df_final.iterrows():
        cid = row["customer_id"]
        status = row["status"]
        cat = row["category"]
        tokens = row["tokens_used"]
        tid = row["ticket_id"]

        is_prem = False
        seg_name = ""
        if cid in seg_map.index:
            is_prem = bool(seg_map.loc[cid, "is_premium"])
            seg_name = str(seg_map.loc[cid, "segment_name"])

        # Routing rules
        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            route = "human_review"
            priority = "NORMAL"
        elif cat == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_prem:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        routed_records.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": cat if status in ["classified_json", "classified_keyword"] else "",
            "segment_name": seg_name if cid in seg_map.index else "",
            "priority": priority,
            "route": route,
            "tokens_used": tokens
        })

    df_triage_results = pd.DataFrame(routed_records)
    out_results_path = os.path.join("outputs", "triage_results.csv")
    df_triage_results.to_csv(out_results_path, index=False)
    print(f"Wrote {len(df_triage_results)} final triage records to {out_results_path}")
    print("Routing Distribution:\n", df_triage_results["route"].value_counts().to_dict())
    print("Priority Distribution:\n", df_triage_results["priority"].value_counts().to_dict())

    print("\nPart C completed successfully.")

if __name__ == "__main__":
    run_part_c()
