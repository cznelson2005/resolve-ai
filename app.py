import streamlit as st
import uuid
import time
from production import (
    TicketState, retrieval_node, evaluation_node, human_review_node,
    response_node, supervisor_node, audit_node, ESCALATION_CONFIG
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="ResolveFlow AI",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================================
# PIPELINE BUILDER
# =====================================================================
@st.cache_resource
def build_pipeline():
    workflow = StateGraph(TicketState)
    workflow.add_node("retrieval",    retrieval_node)
    workflow.add_node("eval",         evaluation_node)
    workflow.add_node("dlq",          human_review_node)
    workflow.add_node("response",     response_node)
    workflow.add_node("supervisor",   supervisor_node)
    workflow.add_node("audit",        audit_node)

    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "eval")

    # Circuit breaker: system_error OR trust_safety → DLQ
    def route_after_eval(state):
        eval_result = state.get("evaluation", {})
        if eval_result.get("issue_type") == "ai_evaluation_failed":
            return "dlq"
        if eval_result.get("trust_safety") is True:
            return "dlq"
        return "response"

    workflow.add_conditional_edges("eval", route_after_eval,
        {"dlq": "dlq", "response": "response"})
    workflow.add_edge("dlq", "audit")

    # Early exit: severity < 4 skips supervisor
    def route_after_response(state):
        return "supervisor" if state.get("evaluation", {}).get("escalate") else "audit"

    workflow.add_conditional_edges("response", route_after_response,
        {"supervisor": "supervisor", "audit": "audit"})
    workflow.add_edge("supervisor", "audit")
    workflow.add_edge("audit", END)

    return workflow.compile(checkpointer=MemorySaver())

pipeline = build_pipeline()

# =====================================================================
# MOCK TOOL CALLS — Simulates Carousell backend lookups
# =====================================================================
def simulate_tool_calls(query: str, user_type: str,
                        transaction_value: float, seller_rating: float,
                        past_disputes: int) -> list[dict]:
    """
    Simulates the tool calls a production system would make
    to Carousell's internal databases before running the pipeline.
    """
    tools_called = []
    q_lower = query.lower()

    # Tool 1: Always look up order if transaction involved
    if transaction_value > 0 or any(w in q_lower for w in
                                     ["order", "item", "payment", "charge", "refund"]):
        tools_called.append({
            "tool"   : "lookup_order",
            "input"  : {"order_id": "ORD-" + uuid.uuid4().hex[:6].upper()},
            "output" : {
                "status"       : "delivered" if "not received" not in q_lower else "in_transit",
                "amount"       : transaction_value or 80,
                "paid_in_app"  : "outside" not in q_lower,
                "delivery_date": "2026-06-18"
            },
            "flag"   : "outside" in q_lower  # flag if off-platform payment suspected
        })

    # Tool 2: Look up seller if buyer complaint or low rating
    if user_type == "buyer" or seller_rating < 4.0:
        tools_called.append({
            "tool"  : "lookup_seller",
            "input" : {"seller_id": "SELLER-" + uuid.uuid4().hex[:4].upper()},
            "output": {
                "rating"         : seller_rating,
                "total_disputes" : past_disputes,
                "account_age_days": 365,
                "active_listings": 12
            },
            "flag"  : seller_rating < ESCALATION_CONFIG["seller_rating_threshold"]
        })

    # Tool 3: Check for duplicate charges if billing issue
    if any(w in q_lower for w in ["charged twice", "double", "duplicate", "billed twice"]):
        tools_called.append({
            "tool"  : "check_payment_records",
            "input" : {"query": query[:40]},
            "output": {
                "duplicate_detected": True,
                "refund_eligible"   : True,
                "duplicate_amount"  : transaction_value or 100
            },
            "flag"  : True  # always a flag
        })

    # Tool 4: Check account status if suspended/locked
    if any(w in q_lower for w in ["suspended", "locked", "banned", "account"]):
        tools_called.append({
            "tool"  : "check_account_status",
            "input" : {"user_type": user_type},
            "output": {
                "status"           : "suspended" if "suspended" in q_lower else "active",
                "suspension_reason": "policy_violation" if "suspended" in q_lower else None,
                "appeal_eligible"  : True
            },
            "flag"  : "suspended" in q_lower
        })

    return tools_called

# =====================================================================
# SESSION STATE INITIALISATION
# =====================================================================
if "thread_id"    not in st.session_state:
    st.session_state.thread_id    = str(uuid.uuid4())
if "conversation" not in st.session_state:
    st.session_state.conversation = []   # list of {role, content, meta}
if "last_state"   not in st.session_state:
    st.session_state.last_state   = None

# =====================================================================
# SIDEBAR — System Config + Carousell Context
# =====================================================================
with st.sidebar:
    st.header("⚙️ System Configuration")

    st.subheader("Escalation Rules")
    escalation_threshold = st.slider(
        "Escalation Threshold (severity ≥)",
        min_value=1, max_value=5, value=4,
        help="Cases at or above this severity will be escalated to supervisor"
    )
    high_value = st.number_input(
        "High Value Transaction ($)",
        min_value=0, max_value=10000, value=200,
        help="Transactions above this amount trigger minimum severity 4 for disputes"
    )
    similarity_threshold = st.slider(
        "Similarity Threshold (past cases)",
        min_value=0.50, max_value=0.99, value=0.81, step=0.01,
        help="Minimum cosine similarity to retrieve a past case"
    )

    # Update live config
    ESCALATION_CONFIG["high_value_threshold"]    = high_value
    ESCALATION_CONFIG["log_similarity_threshold"] = similarity_threshold

    st.divider()

    st.subheader("🛒 Transaction Context")
    user_type = st.selectbox(
        "User Type",
        ["unknown", "buyer", "seller", "platform"],
        help="Carousell user role — affects Trust & Safety routing"
    )
    transaction_value = st.number_input(
        "Transaction Value ($)",
        min_value=0, max_value=10000, value=0,
        help="Higher values increase dispute severity"
    )
    seller_rating = st.slider(
        "Seller Rating",
        min_value=0.0, max_value=5.0, value=5.0, step=0.1,
        help="Ratings below 3.0 trigger Trust & Safety flag"
    )
    past_disputes = st.number_input(
        "Past Disputes",
        min_value=0, max_value=50, value=0,
        help="High dispute count increases severity score"
    )

    st.divider()

    st.subheader("🎬 Demo Scenarios")
    scenario = st.selectbox("Load a scenario", [
        "— select —",
        "Routine Refund Query",
        "Delayed Delivery",
        "Double Charge (Serious)",
        "Off-Platform Payment (Fraud)",
        "Account Suspended (Critical)",
        "Seller Rating Alert"
    ])

    # Reset conversation
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id    = str(uuid.uuid4())
        st.session_state.conversation = []
        st.session_state.last_state   = None
        st.rerun()

# =====================================================================
# SCENARIO PRESETS
# =====================================================================
scenario_presets = {
    "Routine Refund Query": {
        "query": "How long does a refund take?",
        "user_type": "buyer", "transaction_value": 0,
        "seller_rating": 5.0, "past_disputes": 0
    },
    "Delayed Delivery": {
        "query": "My order was supposed to arrive 3 days ago and still nothing",
        "user_type": "buyer", "transaction_value": 80,
        "seller_rating": 4.2, "past_disputes": 0
    },
    "Double Charge (Serious)": {
        "query": "I was charged twice for my order and nobody is helping me. This is unacceptable!",
        "user_type": "buyer", "transaction_value": 150,
        "seller_rating": 4.5, "past_disputes": 1
    },
    "Off-Platform Payment (Fraud)": {
        "query": "The seller is asking me to pay via bank transfer outside the Carousell app",
        "user_type": "buyer", "transaction_value": 300,
        "seller_rating": 2.5, "past_disputes": 0
    },
    "Account Suspended (Critical)": {
        "query": "My account was suspended without any notice and I am losing sales every hour. I want to escalate this immediately.",
        "user_type": "seller", "transaction_value": 0,
        "seller_rating": 4.8, "past_disputes": 0
    },
    "Seller Rating Alert": {
        "query": "Buyer is threatening to leave a bad review unless I give a full refund even though item was delivered",
        "user_type": "seller", "transaction_value": 120,
        "seller_rating": 3.8, "past_disputes": 2
    }
}

# =====================================================================
# MAIN LAYOUT
# =====================================================================
st.title("🛡️ ResolveFlow AI — C2C Support Triage")
st.caption(f"Thread ID: `{st.session_state.thread_id[:16]}...` | "
           f"Turn: {len(st.session_state.conversation) // 2 + 1}")

# ── Two column layout ─────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("💬 Conversation")

    # Display conversation history
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.conversation:
            st.info("Start a conversation below. Use the same thread to simulate multi-turn support.")
        for msg in st.session_state.conversation:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("meta"):
                    meta = msg["meta"]
                    sev  = meta.get("severity_score", "—")
                    esc  = "🚨 Yes" if meta.get("escalate") else "✅ No"
                    st.caption(f"Severity: {sev}/5 | Escalated: {esc} | "
                               f"Latency: {meta.get('total_latency', '—')}s")

    # Input area
    preset_query = ""
    if scenario != "— select —" and scenario in scenario_presets:
        preset = scenario_presets[scenario]
        preset_query = preset["query"]
        # Auto-update sidebar values via session state
        user_type         = preset["user_type"]
        transaction_value = preset["transaction_value"]
        seller_rating     = preset["seller_rating"]
        past_disputes     = preset["past_disputes"]

    query = st.chat_input(
        "Type your support message..." if not preset_query
        else f"Loaded: {preset_query[:50]}... (press enter to run)"
    )

    # Handle preset scenario auto-run
    if scenario != "— select —" and preset_query and not query:
        query = preset_query if st.button(
            f"▶️ Run: {preset_query[:40]}...", use_container_width=True
        ) else None

with right_col:
    st.subheader("📊 Pipeline Results")

    if st.session_state.last_state:
        state = st.session_state.last_state
        evaluation = state.get("evaluation", {})
        supervisor = state.get("supervisor_decision", {})
        metrics    = state.get("latency_metrics", {})
        tools      = state.get("tool_calls_simulated", [])

        # ── Decision Summary ──────────────────────────────────
        sev     = evaluation.get("severity_score", "—")
        sev_map = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🚨"}
        sev_emoji = sev_map.get(sev, "⚪")

        route_label = (
            "🧑 Human Review" if evaluation.get("issue_type") == "ai_evaluation_failed"
            else ("⚡ Auto-Resolved" if not evaluation.get("escalate") else "📢 Escalated")
        )

        total_latency = round(sum(metrics.values()), 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Severity",      f"{sev_emoji} {sev} / 5")
        col2.metric("Escalation",    "Yes" if evaluation.get("escalate") else "No")
        col3.metric("Route",         route_label)
        col4.metric("Total Latency", f"{total_latency}s")

        st.divider()

        # ── Tabs for details ──────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔧 Tool Calls", "📊 Risk Assessment",
            "🗺️ Pipeline Trace", "🧾 Full Ticket"
        ])

        with tab1:
            st.markdown("**Simulated Carousell Backend Lookups**")
            if tools:
                for t in tools:
                    flag_icon = "⚠️" if t.get("flag") else "✅"
                    with st.expander(
                        f"{flag_icon} `{t['tool']}({list(t['input'].values())[0]})`",
                        expanded=t.get("flag", False)
                    ):
                        st.json(t["output"])
                        if t.get("flag"):
                            st.error("🚩 Anomaly detected — flagged for evaluation")
            else:
                st.info("No tool calls triggered for this query.")

        with tab2:
            # Risk assessment
            sentiment_map = {
                "neutral"    : ("😐", "blue"),
                "frustrated" : ("😤", "orange"),
                "angry"      : ("😡", "red")
            }
            sent = evaluation.get("sentiment", "neutral")
            sent_icon, sent_color = sentiment_map.get(sent, ("😐", "blue"))

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Sentiment",   f"{sent_icon} {sent.title()}")
                st.metric("Issue Type",  evaluation.get("issue_type", "—").title())
                st.metric("Repeat Issue", "Yes ⚠️" if evaluation.get("repeat_issue") else "No")
            with col_b:
                st.metric("Trust & Safety", "🛡️ Flagged" if evaluation.get("trust_safety") else "Clear")
                st.metric("Suggested Tone", evaluation.get("suggested_tone", "—").title())

            st.markdown("**AI Reasoning:**")
            st.info(evaluation.get("reasoning", "—"))

            if supervisor.get("action") and supervisor["action"] != "none":
                st.markdown("**Supervisor Decision:**")
                action_map = {
                    "compensate"         : "💰",
                    "escalate_management": "📢",
                    "escalate_rca"       : "🔬",
                    "assign_human"       : "🙋",
                    "monitor"            : "👁️",
                    "none"               : "✅"
                }
                priority_map = {"normal": "🟢", "high": "🟠", "urgent": "🔴"}
                action   = supervisor.get("action", "none")
                priority = supervisor.get("priority", "normal")
                assigned = supervisor.get("assigned_to", "none")
                comp     = supervisor.get("compensation", "none")

                col_c, col_d = st.columns(2)
                col_c.metric("Action",      f"{action_map.get(action, '📋')} {action}")
                col_c.metric("Assigned To", assigned.replace("_", " ").title())
                col_d.metric("Priority",    f"{priority_map.get(priority, '⚪')} {priority.title()}")
                col_d.metric("Compensation", comp.replace("_", " ").title())

                if supervisor.get("internal_notes"):
                    st.caption(f"📝 {supervisor['internal_notes']}")

        with tab3:
            st.markdown("**Agent Execution Trace**")

            steps = [
                ("🔍 Retrieval",  "retrieval_node",  True),
                ("📊 Evaluation", "evaluation_node", True),
                ("💬 Response",   "response_node",
                 evaluation.get("issue_type") != "ai_evaluation_failed"),
                ("🧑‍💼 Supervisor", "supervisor_node",
                 bool(evaluation.get("escalate"))),
                ("📋 Audit",      "audit_node",       True),
            ]

            for name, key, ran in steps:
                latency = metrics.get(key)
                if ran and latency is not None:
                    bar_pct = min(int((latency / max(total_latency, 1)) * 100), 100)
                    st.markdown(
                        f"`{name}` &nbsp; **{latency}s** &nbsp; "
                        f"{'█' * (bar_pct // 10)}{'░' * (10 - bar_pct // 10)} "
                        f"{bar_pct}%"
                    )
                elif not ran:
                    st.markdown(f"`{name}` &nbsp; ⏭️ *skipped (early exit)*")
                else:
                    st.markdown(f"`{name}` &nbsp; ⏭️ *skipped*")

            # Errors
            errors = state.get("errors", [])
            if errors:
                st.markdown("**⚠️ Errors:**")
                for e in errors:
                    st.error(e)

        with tab4:
            ticket_data = {
                "ticket_id"         : state.get("ticket_id", "—"),
                "query"             : state.get("query", ""),
                "severity"          : f"{evaluation.get('severity_score')}/5 — {evaluation.get('severity_label')}",
                "sentiment"         : evaluation.get("sentiment"),
                "issue_type"        : evaluation.get("issue_type"),
                "escalated"         : evaluation.get("escalate"),
                "trust_safety"      : evaluation.get("trust_safety"),
                "action"            : supervisor.get("action"),
                "compensation"      : supervisor.get("compensation"),
                "assigned_to"       : supervisor.get("assigned_to"),
                "priority"          : supervisor.get("priority"),
                "resolution"        : (
                    "human_review" if evaluation.get("issue_type") == "ai_evaluation_failed"
                    else ("pending" if evaluation.get("escalate") else "resolved")
                )
            }
            st.json(ticket_data)
    else:
        st.info("Run the pipeline to see results here.")

# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
if query:
    # Add user message to conversation
    st.session_state.conversation.append({
        "role"   : "user",
        "content": query,
        "meta"   : None
    })

    #Extract previous conversation 
    history_lines = []
    # [:-1] excludes the latest query
    for msg in st.session_state.conversation[:-1]:
        speaker = "Customer" if msg["role"] == "user" else "Agent"
        history_lines.append(f"{speaker}: {msg['content']}")
        
    # extract past 20 histories
    full_history_str = "\n".join(history_lines[-20:]) 
    # For Pinecone indexing（only keep 1-2 sentense for phrase indexing）
    recent_context_str = "\n".join(history_lines[-2:])

    with st.spinner("🤖 Running pipeline..."):
        # Simulate tool calls first (visual effect)
        tool_calls = simulate_tool_calls(
            query, user_type, transaction_value, seller_rating, past_disputes
        )
        time.sleep(0.3)  # brief pause for visual realism

        # Run actual pipeline
        t_start = time.time()
        state   = pipeline.invoke(
            {
                "query"            : query,
                "chat_history"     : full_history_str,  #pass historical records
                "recent_context"   : recent_context_str,
                "user_type"        : user_type,
                "transaction_value": float(transaction_value),
                "seller_rating"    : float(seller_rating),
                "past_disputes"    : int(past_disputes),
                "latency_metrics"  : {},
                "errors"           : []
            },
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )

        # Attach tool calls to state for display
        state["tool_calls_simulated"] = tool_calls
        state["query"]                = query
        total_latency = round(time.time() - t_start, 2)

    # Add assistant response to conversation
    response = state.get("customer_response", "Sorry, something went wrong.")
    evaluation = state.get("evaluation", {})

    st.session_state.conversation.append({
        "role"   : "assistant",
        "content": response,
        "meta"   : {
            "severity_score": evaluation.get("severity_score"),
            "escalate"      : evaluation.get("escalate"),
            "total_latency" : total_latency
        }
    })

    st.session_state.last_state = state
    st.rerun()
