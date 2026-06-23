import streamlit as st
import uuid
from production import *
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. 核心邏輯 (你的原版 build_pipeline，完全不動) ---
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

# --- 2. 介面渲染 (重新架構) ---
st.set_page_config(layout="wide", page_title="ResolveFlow AI")
st.title("🛡️ ResolveFlow AI Dashboard")

# 最上面：Customer Input
query = st.text_area("Customer Ticket", height=100)
run_btn = st.button("🚀 Run Pipeline", type="primary")

# 左右兩欄布局
col_left, col_right = st.columns(2)

if run_btn and query:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"query": query, "latency_metrics": {}}

    # 左邊：執行軌跡 (Trace)
    with col_left:
        st.subheader("🧠 Execution Trace")
        trace_container = st.container(border=True)
        with trace_container:
            # 遍歷執行過程
            for event in app.stream(inputs, config=config):
                for node, data in event.items():
                    st.write(f"✅ Executed: **{node}**")
                    if node == "evaluation_node":
                        st.json(data.get("evaluation", {}))
                    if node == "supervisor_node":
                        st.json(data.get("supervisor_decision", {}))

    # 右邊：最終回覆 (Output)
    with col_right:
        st.subheader("💬 Customer Response")
        # 獲取最終狀態
        final_state = app.get_state(config).values
        
        # 這裡會等到所有執行完成才顯示
        if "customer_response" in final_state:
            st.info(final_state["customer_response"])
            
            st.subheader("📊 Telemetry")
            st.json(final_state.get("latency_metrics", {}))
            
            if "supervisor_decision" in final_state:
                st.warning(f"Supervisor Action: {final_state['supervisor_decision'].get('action')}")
