import streamlit as st
import uuid
from production import *
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Pipeline Definition
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

# 2. UI Rendering
st.title("🛡️ ResolveFlow AI")
query = st.text_area("Customer Ticket")
if st.button("🚀 Run Pipeline"):
    state = app.invoke({"query": query, "latency_metrics": {}}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    
    # Render node results
    st.success(f"Customer Response: {state.get('customer_response')}")
    st.subheader("Telemetry")
    st.json(state.get("latency_metrics"))
    if "supervisor_decision" in state:
        st.warning(f"Supervisor Action: {state['supervisor_decision'].get('action')}")