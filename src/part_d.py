"""src/part_d.py - Policy RAG Assistant and Refund Agent
Implements chunking, TF-IDF vector retrieval, RAG evaluation (overlap vs zero-overlap),
and a resilient, budget-bounded, resumable ReAct refund agent.
"""

import os
import sys
import json
import time
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath("."))
import mock_llm

POL_DIR = os.path.join("data", "policies")
DB_PATH = os.path.join("data", "instamart.db")
LEDGER_PATH = os.path.join("outputs", "refund_ledger.jsonl")
RUNS_PATH = os.path.join("outputs", "agent_runs.jsonl")
STATE_DIR = "state"

def create_chunks(overlap=10):
    """Splits policy documents into 40-word chunks with specified overlap."""
    files = sorted(os.listdir(POL_DIR))
    chunks = []
    chunk_size = 40
    step = chunk_size - overlap
    for fn in files:
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(POL_DIR, fn), "r", encoding="utf-8") as f:
            words = f.read().split()
        i = 0
        while True:
            c = " ".join(words[i:i + chunk_size])
            chunks.append({"source": fn, "text": c})
            if i + chunk_size >= len(words):
                break
            i += step
    return chunks

# Chunks and Vectorizer with Overlap=10
chunks_10 = create_chunks(overlap=10)
vectorizer_10 = TfidfVectorizer()
tfidf_matrix_10 = vectorizer_10.fit_transform([c["text"] for c in chunks_10])

def retrieve(query, k=3, vectorizer=vectorizer_10, tfidf_matrix=tfidf_matrix_10, chunks=chunks_10):
    """Ranks chunks by cosine similarity to the query TF-IDF vector."""
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    ranked = sims.argsort()[::-1][:k]
    return [{"source": chunks[i]["source"], "score": float(sims[i]), "text": chunks[i]["text"]} for i in ranked]

def answer_policy_question(question, retrieve_fn=retrieve):
    """Augments prompt with retrieved policy chunks and queries mock_llm."""
    retrieved = retrieve_fn(question, k=3)
    context_blocks = [f"[{r['source']}]\n{r['text']}" for r in retrieved]
    context_str = "\n\n".join(context_blocks)
    
    sys_msg = (
        "You are a POLICY ASSISTANT. Answer the user question ONLY from the supplied CONTEXT. "
        "If the answer cannot be found in the context, reply exactly with: "
        "I could not find that information in the policy documents."
    )
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
    
    resp = mock_llm.chat([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg}
    ])
    return resp.choices[0].message.content, [r["source"] for r in retrieved]

# --- Refund Agent Tools & Specifications ---

def get_order(order_id):
    """Reads order details from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": int(row[3]) if row[3] is not None and str(row[3]).strip() != "" and not pd.isna(row[3]) else None
    }

def search_policy(query):
    """Searches store policies using TF-IDF retrieval."""
    return {"results": retrieve(query, k=3)}

def calculate_refund(order_amount, delivery_minutes):
    """Calculates refund based on late delivery policy (>45 mins)."""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}

def issue_refund(order_id, refund_inr):
    """Issues automatic refund for amounts <= INR 1000 and logs to refund_ledger.jsonl."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    """Escalates high-value refunds (> INR 1000) to human lead and logs to refund_ledger.jsonl."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Reads order status, amount, and delivery time from instamart database.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Searches store policy documents using TF-IDF retrieval.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculates refundable amount based on order delivery time and amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "integer"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issues automatic refund for amounts up to INR 1000.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalates high-value refund requests (> INR 1000) to a human lead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]

def save_agent_checkpoint(run_id, scenario_id, messages, tokens_used, step, tools_executed):
    """Saves agent execution state to state/<run_id>.json."""
    os.makedirs(STATE_DIR, exist_ok=True)
    ckpt = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step": step,
        "tools_executed": tools_executed
    }
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_tool=None):
    """Executes ReAct refund loop with safety guardrails and checkpoints."""
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time()*1000)}"
    start_time = time.monotonic()
    
    messages = [
        {"role": "system", "content": "You are a customer support refund AGENT. Use tools to verify orders, search policies, and calculate and process refunds."},
        {"role": "user", "content": user_message}
    ]
    
    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []
    
    for iteration in range(1, max_iterations + 1):
        if time.monotonic() - start_time >= time_limit_seconds:
            summary = f"Timeout reached after {iteration-1} steps. Handoff to human agent."
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT",
                "final_answer": summary, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        
        if tokens_used > token_limit:
            summary = f"Token budget of {token_limit} exceeded at call {llm_calls}. Handoff to human agent."
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": summary, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        choice = resp.choices[0]
        if not choice.message.tool_calls:
            final_content = choice.message.content or ""
            status = final_content.split(":")[0].strip() if ":" in final_content else "UNKNOWN"
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": status,
                "final_answer": final_content, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        tc = choice.message.tool_calls[0]
        tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]
        })
        
        tool_func = TOOL_MAP[tc.name]
        tool_result = tool_func(**tool_args)
        tools_executed.append(tc.name)
        
        steps.append({
            "step": iteration,
            "tool": tc.name,
            "arguments": tool_args,
            "observation": tool_result
        })
        
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(tool_result)
        })
        
        save_agent_checkpoint(run_id, scenario_id, messages, tokens_used, iteration, tools_executed)
        
        if crash_after_tool and len(tools_executed) == crash_after_tool:
            raise RuntimeError(f"Simulated crash after executing tool #{crash_after_tool}: {tc.name}")
            
    summary = f"Reached maximum allowed iterations ({max_iterations}) without resolution. Handoff to human agent."
    return {
        "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": summary, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    """Resumes agent execution from state/<run_id>.json checkpoint."""
    start_time = time.monotonic()
    ckpt_path = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    scenario_id = data["scenario_id"]
    messages = data["messages"]
    tokens_used = data["tokens_used"]
    tools_executed = list(data["tools_executed"])
    steps = []
    for m in messages:
        if m.get("role") == "tool":
            steps.append({
                "tool": m["name"],
                "observation": json.loads(m["content"])
            })
            
    current_step = data["step"]
    llm_calls = 0
    
    for iteration in range(current_step + 1, max_iterations + 1):
        if time.monotonic() - start_time >= time_limit_seconds:
            summary = "Timeout reached during resumed run. Handoff to human agent."
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT",
                "final_answer": summary, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        
        if tokens_used > token_limit:
            summary = f"Token budget of {token_limit} exceeded during resumed run. Handoff to human agent."
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": summary, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        choice = resp.choices[0]
        if not choice.message.tool_calls:
            final_content = choice.message.content or ""
            status = final_content.split(":")[0].strip() if ":" in final_content else "UNKNOWN"
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": status,
                "final_answer": final_content, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        tc = choice.message.tool_calls[0]
        tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]
        })
        
        tool_func = TOOL_MAP[tc.name]
        tool_result = tool_func(**tool_args)
        tools_executed.append(tc.name)
        
        steps.append({
            "step": iteration,
            "tool": tc.name,
            "arguments": tool_args,
            "observation": tool_result
        })
        
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(tool_result)
        })
        
        save_agent_checkpoint(run_id, scenario_id, messages, tokens_used, iteration, tools_executed)
        
    summary = f"Reached maximum allowed iterations ({max_iterations}) without resolution. Handoff to human agent."
    return {
        "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": summary, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }

def run_part_d():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    print("==================================================================")
    print("PART D: POLICY RAG ASSISTANT AND REFUND AGENT")
    print("==================================================================")

    # Task 1: Chunking Checkpoints
    print("\n--- Task 1: Document Chunking ---")
    print(f"Total chunks (overlap=10): {len(chunks_10)}")
    doc_chunk_counts = pd.Series([c["source"] for c in chunks_10]).value_counts().to_dict()
    print("Chunks per policy document:", doc_chunk_counts)

    # Task 2: Vectorise and Retrieve
    print("\n--- Task 2: TF-IDF Retrieval Test ---")
    top_chunks = retrieve("refund policy late delivery", k=3)
    print("Top 3 retrieved chunks for 'refund policy late delivery':")
    for idx, r in enumerate(top_chunks, 1):
        print(f"  Rank {idx} [{r['source']}]: similarity={r['score']:.4f}")

    # Task 4: Evaluation of Grounding (Overlap 10 vs Overlap 0)
    print("\n--- Task 4: Policy RAG Evaluation (policy_eval.csv) ---")
    eval_df = pd.read_csv(os.path.join("data", "raw", "policy_eval.csv"))

    # Evaluation with Overlap=10
    top_doc_match_10 = 0
    fact_match_10 = 0
    fallback_match_10 = 0

    for _, row in eval_df.iterrows():
        q, exp_doc, exp_fact = row["question"], row["expected_doc"], row["expected_fact"]
        ans, sources = answer_policy_question(q)
        if exp_doc != "NONE":
            if exp_doc in sources:
                top_doc_match_10 += 1
            if exp_fact in ans:
                fact_match_10 += 1
        else:
            if ans.startswith("I could not find"):
                fallback_match_10 += 1

    print("Overlap=10 Results:")
    print(f"  Top-3 Doc Match: {top_doc_match_10} / 12")
    print(f"  Expected Fact Match: {fact_match_10} / 12")
    print(f"  Unanswerable Fallback Match: {fallback_match_10} / 2")

    # Evaluation with Overlap=0
    chunks_0 = create_chunks(overlap=0)
    vectorizer_0 = TfidfVectorizer()
    tfidf_matrix_0 = vectorizer_0.fit_transform([c["text"] for c in chunks_0])

    def retrieve_0(q, k=3):
        return retrieve(q, k=k, vectorizer=vectorizer_0, tfidf_matrix=tfidf_matrix_0, chunks=chunks_0)

    top_doc_match_0 = 0
    fact_match_0 = 0
    fallback_match_0 = 0
    failed_questions_0 = []

    for _, row in eval_df.iterrows():
        q, exp_doc, exp_fact = row["question"], row["expected_doc"], row["expected_fact"]
        ans, sources = answer_policy_question(q, retrieve_fn=retrieve_0)
        if exp_doc != "NONE":
            if exp_doc in sources:
                top_doc_match_0 += 1
            if exp_fact in ans:
                fact_match_0 += 1
            else:
                failed_questions_0.append((q, exp_fact, ans))
        else:
            if ans.startswith("I could not find"):
                fallback_match_0 += 1

    print("\nOverlap=0 Results:")
    print(f"  Total chunks: {len(chunks_0)}")
    print(f"  Top-3 Doc Match: {top_doc_match_0} / 12")
    print(f"  Expected Fact Match: {fact_match_0} / 12")
    print(f"  Unanswerable Fallback Match: {fallback_match_0} / 2")
    print(f"  Failed Questions under Overlap=0 ({len(failed_questions_0)}):")
    for fq, fact, got in failed_questions_0:
        print(f"    - Question: '{fq}'\n      Expected Fact: '{fact}'\n      Received Answer: '{got}'")

    # Task 6: Run 7 Scenarios
    print("\n--- Task 6: ReAct Agent Runs (7 Scenarios) ---")
    # Reset output files
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)

    scenarios_df = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"))
    scenario_results = []

    for _, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        exp_oc = row["expected_outcome"]
        exp_amt = float(row["expected_refund_inr"])

        res = run_agent(msg, sid)
        
        # Extract refund amount if any
        refund_amt = 0.0
        if "INR" in res["final_answer"]:
            try:
                refund_amt = float(res["final_answer"].split("INR")[1].split()[0])
            except Exception:
                pass

        scenario_results.append({
            "scenario_id": sid,
            "status": res["status"],
            "expected_outcome": exp_oc,
            "refund_amount": refund_amt,
            "expected_refund_inr": exp_amt,
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "final_answer": res["final_answer"],
            "steps": res["steps"]
        })

        # Append to outputs/agent_runs.jsonl
        run_log_entry = {
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        }
        with open(RUNS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_log_entry) + "\n")

        print(f"  {sid}: status={res['status']:22} calls={res['llm_calls']} tokens={res['tokens_used']:4} answer='{res['final_answer']}'")

    # Task 7: Stop Safely Demonstrations
    print("\n--- Task 7: Stop Safely Guardrail Demonstrations ---")
    s1_msg = scenarios_df.iloc[0]["message"]
    
    # Demo 1: Token limit fallback
    res_token = run_agent(s1_msg, "S1_demo_token", token_limit=200)
    print(f"  Token limit demonstration (token_limit=200) -> Status: {res_token['status']}")
    
    # Demo 2: Timeout fallback
    res_timeout = run_agent(s1_msg, "S1_demo_timeout", time_limit_seconds=0)
    print(f"  Timeout demonstration (time_limit_seconds=0) -> Status: {res_timeout['status']}")

    # Task 8: Resumable Execution Demonstration
    print("\n--- Task 8: Resumable Execution Demonstration (S1) ---")
    crash_run_id = "s1_resumable_demo"
    try:
        run_agent(s1_msg, "S1", run_id=crash_run_id, crash_after_tool=2)
    except RuntimeError as e:
        print(f"  Simulated crash triggered: {e}")

    resumed_result = resume_agent(crash_run_id)
    print(f"  Resumed Run Status: {resumed_result['status']}")
    print(f"  Final Answer: {resumed_result['final_answer']}")
    print(f"  Tools executed across crash + resume ({len(resumed_result['tools_executed'])}): {resumed_result['tools_executed']}")
    assert resumed_result["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"], "Tool sequence mismatch across crash-resume."

    print("\nPart D completed successfully.")

if __name__ == "__main__":
    run_part_d()
