import streamlit as st
import uuid
import time
from production import *
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- Configuration & Pipeline ---
# 你的 build_pipeline 邏輯保持不變
def build_pipeline():
    workflow = StateGraph(TicketState)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("eval", evaluation_node)
    workflow.add_node("dlq", human_review_node)
    workflow.add_node("response", response_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("audit", audit_node)
    
    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "eval")
    workflow.add_conditional_edges("eval", lambda s: "dlq" if s.get("evaluation", {}).get("issue_type") == "ai_evaluation_failed" else "response")
    workflow.add_edge("dlq", "audit")
    workflow.add_conditional_edges("response", lambda s: "supervisor" if s.get("evaluation", {}).get("escalate") else "audit")
    workflow.add_edge("supervisor", "audit")
    workflow.add_edge("audit", END)
    return workflow.compile(checkpointer=MemorySaver())

app = build_pipeline()

# --- UI Layout ---
st.set_page_config(layout="wide", page_title="ResolveFlow AI")
st.title("🛡️ ResolveFlow AI — Support Triage Simulator")

# 1. Scenarios
SCENARIOS = {
    "Routine Refund Question": "How do I update my billing address?",
    "Low-Value Delivery Dispute": "The item I received is slightly different from the listing. It costs $12.",
    "High-Value Watch Scam": "I had a chat with a seller regarding buying a $1000 watch. I made a $500 payment and the seller disappeared.",
    "Account Takeover": "I think someone hacked my account and changed my payout details.",
    "Off-Platform Payment Attempt": "The seller asked me to transfer money via bank transfer outside the platform."
}

col_top1, col_top2 = st.columns([1, 3])
with col_top1:
    scenario = st.selectbox("Demo Scenario", list(SCENARIOS.keys()))
with col_top2:
    query = st.text_area("Customer Ticket", value=SCENARIOS[scenario], height=100)

if st.button("🚀 Run Pipeline", type="primary"):
    thread_id = str(uuid.uuid4())
    inputs = {"query": query, "latency_metrics": {}}
    
    # Run pipeline
    final_state = app.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
    
    # 2. Executive Metrics Cards
    eval_data = final_state.get("evaluation", {})
    sup_data = final_state.get("supervisor_decision", {})
    latency = final_state.get("latency_metrics", {})
    
    severity = eval_data.get("severity_score", "N/A")
    escalate = eval_data.get("escalate", False)
    route = "Supervisor Review" if escalate else "Auto Response"
    total_lat = sum(latency.values())
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Severity", f"{severity}/5")
    m2.metric("Escalation", "Yes" if escalate else "No")
    m3.metric("Route", route)
    m4.metric("Total Latency", f"{total_lat:.2f}s")
    
    st.divider()

    # 3. Layout: Two Columns
    col1, col2 = st.columns(2)

    with col1:
        # Execution Trace
        st.subheader("🧠 Execution Trace")
        # 直接顯示執行路徑
        steps = ["retrieval", "eval"]
        if eval_data.get("issue_type") == "ai_evaluation_failed": steps.append("dlq")
        else: 
            steps.append("response")
            if escalate: steps.append("supervisor")
        steps.append("audit")
        st.caption(" → ".join([s.upper() for s in steps]))

        # Risk Signals
        st.subheader("🚨 Risk Signals")
        signals = []
        if "$500" in query or "$1000" in query: signals.append("High transaction value")
        if "bank" in query.lower() or "payment" in query.lower(): signals.append("Payment-related risk")
        if "disappeared" in query.lower() or "hacked" in query.lower(): signals.append("Fraud pattern detected")
        if eval_data.get("trust_safety"): signals.append("Trust & Safety Violation")
        
        for s in signals: st.warning(s)

        # Evidence
        with st.expander("🔍 Retrieved Evidence"):
            st.write("**Policies:**", final_state.get("support_contexts", []))
            st.write("**Historical Logs:**", final_state.get("past_cases", []))

    with col2:
        # Customer Response
        st.subheader("💬 Customer Response")
        st.info(final_state.get("customer_response", "N/A"))
        
        # Supervisor Decision
        if escalate:
            st.subheader("🧑‍💼 Supervisor Decision")
            st.warning(f"**Action:** {sup_data.get('action')}")
            st.write(f"**Notes:** {sup_data.get('internal_notes')}")
        
        # Telemetry JSON
        with st.expander("📊 Full Telemetry State"):
            st.json(final_state)
