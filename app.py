import streamlit as st
import uuid
import pandas as pd
from production import *
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. Pipeline Build (與您的 production.py 對接) ---
def build_pipeline():
    workflow = StateGraph(TicketState)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("eval", evaluation_node)
    workflow.add_node("response", response_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("audit", audit_node)
    
    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "eval")
    workflow.add_conditional_edges("eval", lambda s: "response" if not s.get("evaluation", {}).get("escalate") else "supervisor")
    workflow.add_edge("response", "audit")
    workflow.add_edge("supervisor", "audit")
    workflow.add_edge("audit", END)
    return workflow.compile(checkpointer=MemorySaver())

app = build_pipeline()

# --- 2. 監控儀表板函式 ---
def render_telemetry(final_state):
    evaluation = final_state.get("evaluation", {})
    supervisor = final_state.get("supervisor_decision", {})
    latency = final_state.get("latency_metrics", {})
    
    severity_score = evaluation.get("severity_score", "N/A")
    escalate = evaluation.get("escalate", False)
    route = "Supervisor Review" if escalate else "Auto-Resolved"
    total_latency = round(sum(latency.values()), 2) if latency else 0

    st.subheader("📊 Decision Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Severity", f"{severity_score} / 5")
    m2.metric("Escalation", "Yes" if escalate else "No")
    m3.metric("Route", route)
    m4.metric("Total Latency", f"{total_latency}s")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("🚨 Risk Assessment")
        if evaluation.get("reasoning"): st.warning(evaluation.get("reasoning"))
    with right:
        st.subheader("🧑‍💼 Supervisor Decision")
        if supervisor: st.info(f"Action: {supervisor.get('action')}")
        else: st.success("Auto-resolved.")

    st.subheader("📚 Retrieved Evidence")
    if final_state.get("support_contexts"):
        for ctx in final_state["support_contexts"][:2]: 
            with st.expander("Policy Match"): st.write(ctx.get("instruction", ""))
    else:
        st.caption("⚠️ No relevant policies found (Similarity score below threshold).")

# --- 3. UI 渲染 ---
st.set_page_config(layout="wide", page_title="ResolveFlow AI")
st.title("🛡️ ResolveFlow AI — Support Triage Simulator")

# Sidebar Configuration
st.sidebar.subheader("⚙️ System Configuration")
sim_threshold = st.sidebar.slider(
    "Similarity Threshold", 
    min_value=0.50, 
    max_value=0.99, 
    value=0.70,
    step=0.01
)

SCENARIOS = {
    "High-Value Watch Scam": "I made a $500 payment and the seller disappeared.",
    "Routine Refund": "How long does a refund take?",
    "Off-Platform Payment": "The seller asked me to transfer money via bank."
}

col_top1, col_top2 = st.columns([1, 3])
scenario = col_top1.selectbox("Demo Scenario", list(SCENARIOS.keys()))
query = col_top2.text_area("Customer Ticket", value=SCENARIOS[scenario], height=100)

if st.button("🚀 Run Pipeline", type="primary"):
    thread_id = str(uuid.uuid4())
    # 將 threshold 作為 state 的一部分傳入
    inputs = {"query": query, "latency_metrics": {}, "threshold": sim_threshold}
    final_state = app.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
    
    render_telemetry(final_state)
