import streamlit as st
import time
import uuid
import json

# Import V3-aligned state and nodes from production.py
from production import (
    TicketState, 
    retrieval_node, 
    evaluation_node, 
    human_review_node,
    response_node, 
    supervisor_node, 
    audit_node
)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# =====================================================================
# 1. GRAPH COMPILATION (STRICT ALIGNMENT WITH NOTEBOOK V3)
# =====================================================================
def route_after_evaluation(state: TicketState) -> str:
    """Circuit Breaker: Route to DLQ if AI fails or Trust & Safety risk is high."""
    eval_result = state.get("evaluation", {})
    
    if eval_result.get("issue_type") == "ai_evaluation_failed":
        return "human_review_node"
    if eval_result.get("trust_safety") is True:
        return "human_review_node"
        
    return "response_node"

def route_after_response(state: TicketState) -> str:
    """Early Exit: If escalation flag is True, route to supervisor; else skip to audit."""
    eval_result = state.get("evaluation", {})
    if eval_result.get("escalate") is True:
        return "supervisor_node"
    return "audit_node"

@st.cache_resource
def build_pipeline():
    """Compiles the LangGraph workflow using exact V3 notebook node configurations."""
    workflow = StateGraph(TicketState)
    
    # Register all nodes matching production.py
    workflow.add_node("retrieval_node",    retrieval_node)
    workflow.add_node("evaluation_node",   evaluation_node)
    workflow.add_node("human_review_node", human_review_node)
    workflow.add_node("response_node",     response_node)
    workflow.add_node("supervisor_node",   supervisor_node)
    workflow.add_node("audit_node",        audit_node)
    
    # Define exact edge connections
    workflow.set_entry_point("retrieval_node")
    workflow.add_edge("retrieval_node", "evaluation_node")
    
    workflow.add_conditional_edges(
        "evaluation_node",
        route_after_evaluation,
        {
            "human_review_node": "human_review_node",
            "response_node"    : "response_node"
        }
    )
    
    workflow.add_edge("human_review_node", "audit_node")
    
    workflow.add_conditional_edges(
        "response_node",
        route_after_response,
        {
            "supervisor_node": "supervisor_node",
            "audit_node"     : "audit_node"
        }
    )
    
    workflow.add_edge("supervisor_node", "audit_node")
    workflow.add_edge("audit_node", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

app = build_pipeline()

# =====================================================================
# 2. STREAMLIT UI LAYOUT & INTERFACE
# =====================================================================
st.set_page_config(page_title="ResolveFlow AI Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ ResolveFlow AI")
st.subheader("Multi-Agent Trust & Safety Triage System Control Panel")

# --- SIDEBAR: C2C Marketplace Context Simulation ---
with st.sidebar:
    st.header("⚙️ Transaction Context")
    st.markdown("Simulate metadata passed from the C2C marketplace backend.")
    
    user_type = st.selectbox("User Type", ["buyer", "seller", "platform", "unknown"])
    transaction_value = st.slider("Transaction Value ($)", 0.0, 1000.0, 150.0)
    seller_rating = st.slider("Seller Rating (0-5)", 0.0, 5.0, 4.2)
    past_disputes = st.number_input("Past Disputes Count", min_value=0, max_value=10, value=0)
    payment_method = st.selectbox("Payment Method", ["in_app", "off_platform_bank_transfer", "credit_card"])
    
    st.markdown("---")
    st.info("💡 Pro-Tip for Demo: Try inputting an off-platform payment query or lowering the seller rating below 3.0 to watch the Circuit Breaker trigger instantly.")

# --- MAIN INTERACTION LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 Customer Ticket")
    query = st.text_area(
        "Customer Complaint / Inquiry", 
        height=150, 
        placeholder="Type a complaint here (e.g., 'I was charged twice...' or 'The seller wants me to transfer money via bank...')"
    )
    run_btn = st.button("🚀 Run Multi-Agent Pipeline", use_container_width=True, type="primary")

with col2:
    st.header("🧠 Agent Execution Trace")
    
    if run_btn and query:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Construct the verified initial state input dictionary
        inputs = {
            "query"            : query,
            "user_type"        : user_type,
            "transaction_value": transaction_value,
            "seller_rating"    : seller_rating,
            "past_disputes"    : past_disputes,
            "payment_method"   : payment_method,
            "messages"         : [],
            "latency_metrics"  : {},
            "errors"           : []
        }
        
        # Display live streaming nodes as they complete processing
        with st.status("Orchestrating Agents...", expanded=True) as status:
            start_time = time.time()
            
            for event in app.stream(inputs, config=config):
                for node_name, node_data in event.items():
                    st.write(f"✅ Finished: `{node_name}`")
                    
                    # Collapsible live data peeking
                    if node_name == "evaluation_node" and "evaluation" in node_data:
                        st.json(node_data["evaluation"])
                    if node_name == "supervisor_node" and "supervisor_decision" in node_data:
                        st.json(node_data["supervisor_decision"])
                        
            total_time = round(time.time() - start_time, 2)
            status.update(label=f"Pipeline Completed in {total_time}s", state="complete", expanded=False)
            
        # --- RESULTS & METRICS DISPLAY ---
        final_state = app.get_state(config).values
        eval_metrics = final_state.get("evaluation", {})
        latency_data = final_state.get("latency_metrics", {})
        
        st.subheader("🎯 System Resolution Output")
        
        # Visual routing flag diagnostics
        if eval_metrics.get("severity_label") == "system_error" or eval_metrics.get("issue_type") == "ai_evaluation_failed":
            st.error("🚨 CIRCUIT BREAKER CRITICAL TRIPPED: AI Engine exception occurred. Safely rerouted to Human Review Queue (DLQ).")
        elif eval_metrics.get("trust_safety") is True:
            st.error("🛡️ TRUST & SAFETY BREACH: High-risk policy violation detected. Short-circuited directly to Fraud Team.")
        elif "supervisor_node" not in latency_data:
            st.success("⚡ PERFORMANCE OPTIMIZATION: Early Exit triggered for Low Severity (Severity 1-2). Supervisor Agent bypassed to minimize latency and token costs.")
        else:
            st.info("🔍 ESCALATION PATH: Complex case. Multi-layered validation completed via Supervisor review.")
            
        # Output Text Fields
        st.markdown("**Customer-Facing Response:**")
        st.info(final_state.get("customer_response", "No response text generated."))
        
        if final_state.get("ticket_id"):
            st.caption(f"**Ticket Reference ID:** {final_state.get('ticket_id')} (Logged into Pinecone)")

        # Telemetry Summary Block
        st.markdown("**⏱️ Telemetry & Performance Metrics:**")
        st.json(latency_data)
        
        if final_state.get("errors"):
            st.markdown("**⚠️ Logged Fallback Warnings:**")
            st.warning(final_state.get("errors"))