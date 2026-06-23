import streamlit as st
import uuid
import pandas as pd
from production import *
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- Pipeline Logic (保持不變) ---
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

# --- 決策監控儀表板函式 ---
def render_telemetry(final_state):
    evaluation = final_state.get("evaluation", {})
    supervisor = final_state.get("supervisor_decision", {})
    latency = final_state.get("latency_metrics", {})
    support_contexts = final_state.get("support_contexts", [])
    past_cases = final_state.get("past_cases", [])
    errors = final_state.get("errors", [])

    severity_score = evaluation.get("severity_score", "N/A")
    severity_label = evaluation.get("severity_label", "N/A")
    issue_type = evaluation.get("issue_type", "N/A")
    trust_safety = evaluation.get("trust_safety", False)
    escalate = evaluation.get("escalate", False)

    route = "Supervisor Review" if escalate else "Auto-Resolved"
    total_latency = round(sum(latency.values()), 2) if latency else 0

    st.subheader("📊 Decision Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Severity", f"{severity_score} / 5", severity_label)
    m2.metric("Escalation", "Yes" if escalate else "No")
    m3.metric("Route", route)
    m4.metric("Total Latency", f"{total_latency}s")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("🚨 Risk Assessment")
        risk_items = [("Issue Type", issue_type), ("Sentiment", evaluation.get("sentiment", "N/A")), ("Escalation", "Yes" if escalate else "No")]
        for label, value in risk_items: st.write(f"**{label}:** {value}")
        if evaluation.get("reasoning"): st.warning(evaluation.get("reasoning"))

    with right:
        st.subheader("🧑‍💼 Supervisor Decision")
        if supervisor:
            st.write(f"**Action:** `{supervisor.get('action', 'N/A')}`")
            st.write(f"**Assigned To:** `{supervisor.get('assigned_to', 'N/A')}`")
            st.info(supervisor.get("internal_notes", "No notes."))
        else:
            st.success("Auto-resolved path.")

    st.divider()
    st.subheader("⏱️ Node Latency")
    if latency:
        latency_df = pd.DataFrame([{"Node": k.replace("_node", "").replace("_", " ").title(), "Latency (s)": v} for k, v in latency.items()])
        st.bar_chart(latency_df.set_index("Node"))
    
    st.subheader("📚 Retrieved Evidence")
    e1, e2 = st.columns(2)
    with e1:
        st.write("**Support Policy Matches**")
        # 檢查是否為空，若空則顯示提示
        if final_state.get("support_contexts"):
            for i, ctx in enumerate(final_state["support_contexts"][:3], start=1):
                with st.expander(f"{i}. {ctx.get('intent', 'Policy')}"):
                    st.write(f"**Source:** `{ctx.get('source', 'N/A')}`")
                    st.write(ctx.get("response", "No response text available."))
        else:
            st.caption("⚠️ No relevant policies found (Similarity score below threshold).")
    with e2:
        st.write("**Historical Case Matches**")
        # 檢查是否為空
        if final_state.get("past_cases"):
            for i, case in enumerate(final_state["past_cases"][:3], start=1):
                with st.expander(f"Historical Case {i} (Score: {case.get('score')})"):
                    st.json(case)
        else:
            st.caption("⚠️ No relevant historical cases found.")

# --- UI 渲染 ---
st.set_page_config(layout="wide", page_title="ResolveFlow AI")
st.title("🛡️ ResolveFlow AI — Support Triage Simulator")

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
    final_state = app.invoke({"query": query, "latency_metrics": {}}, config={"configurable": {"thread_id": thread_id}})
    
    # 這裡顯示最終報告
    render_telemetry(final_state)
