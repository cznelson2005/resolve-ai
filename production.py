import os
import time
import json
import uuid
import datetime
from typing import TypedDict, Dict, Any, List, Annotated
from pydantic import BaseModel, Field

# Import LangChain and Gemini related packages
import google.genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pinecone import Pinecone

# =====================================================================
# 0. INITIALIZE API CLIENTS & CONFIG
# =====================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize Native GenAI Client for Embeddings
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize LangChain LLM
lc_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GEMINI_API_KEY
)

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "customer-support-rag"
NAMESPACE_DOCS = "support-docs"
NAMESPACE_LOGS = "action-logs"
pinecone_index = pc.Index(INDEX_NAME)

# Escalation config
ESCALATION_CONFIG = {
    "past_dispute_threshold"  : 2,     
    "seller_rating_threshold" : 3.0,   
    "high_value_threshold"    : 200,   
    "log_similarity_threshold": 0.81,  
}

# =====================================================================
# 1. STATE DEFINITION & STRUCTURED OUTPUT SCHEMAS
# =====================================================================
class TicketState(TypedDict, total=False):
    query             : str
    chat_history      : str        
    recent_context    : str        
    user_type         : str        
    transaction_value : float
    seller_rating     : float
    past_disputes     : int
    
    support_contexts  : List[Dict[str, Any]]
    past_cases        : List[Dict[str, Any]]
    evaluation        : Dict[str, Any]
    customer_response : str
    supervisor_decision: Dict[str, Any]
    ticket_id         : str
    
    messages          : Annotated[list[BaseMessage], add_messages]
    
    latency_metrics   : Dict[str, float]
    token_metrics     : Dict[str, int]   
    errors            : List[str]

class EvaluationSchema(BaseModel):
    severity_score : int  = Field(ge=1, le=5, description="1=routine to 5=critical")
    severity_label : str  = Field(description="routine|minor|moderate|serious|critical")
    sentiment      : str  = Field(description="neutral|frustrated|angry")
    issue_type     : str  = Field(description="billing|delivery|account|refund|dispute|fraud|other")
    repeat_issue   : bool
    escalate       : bool
    trust_safety   : bool = Field(default=False, description="Flag for Trust & Safety team")
    reasoning      : str  = Field(description="max 10 words explaining the score")
    suggested_tone : str  = Field(description="professional|empathetic|urgent")

class SupervisorSchema(BaseModel):
    escalated      : bool
    action         : str = Field(description="compensate|escalate_management|escalate_rca|assign_human|monitor|none")
    compensation   : str = Field(description="none|10%_discount|20%_discount|full_refund|service_credit|free_month")
    assigned_to    : str = Field(description="none|human_agent|senior_manager|rca_team|trust_safety_team|legal_team")
    priority       : str = Field(description="normal|high|urgent")
    internal_notes : str = Field(description="max 10 words")

# Ensure include_raw=True is set to extract token usage metadata
eval_llm = lc_llm.with_structured_output(EvaluationSchema, include_raw=True)
supervisor_llm = lc_llm.with_structured_output(SupervisorSchema, include_raw=True)

# =====================================================================
# 2. EMBEDDING FUNCTIONS
# =====================================================================
def embed_query(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"task_type": "RETRIEVAL_QUERY"}
    )
    return response.embeddings[0].values

# =====================================================================
# 2.5 HELPER FUNCTIONS
# =====================================================================
def _extract_tokens(ai_msg) -> tuple[int, int, int]:
    """
    Robustly extract tokens from LangChain AIMessage across different versions.
    Returns: (input_tokens, output_tokens, total_tokens)
    """
    # Method 1: LangChain standard usage_metadata
    if hasattr(ai_msg, "usage_metadata") and ai_msg.usage_metadata:
        um = ai_msg.usage_metadata
        return (
            um.get("input_tokens", 0),
            um.get("output_tokens", 0),
            um.get("total_tokens", 0)
        )
    # Method 2: Gemini-specific response_metadata (camelCase keys)
    if hasattr(ai_msg, "response_metadata") and ai_msg.response_metadata:
        usage = ai_msg.response_metadata.get("usageMetadata", {})
        return (
            usage.get("promptTokenCount", 0),       # ← camelCase
            usage.get("candidatesTokenCount", 0),   # ← camelCase
            usage.get("totalTokenCount", 0)         # ← camelCase
        )
    return 0, 0, 0

# =====================================================================
# 3. AGENT NODES
# =====================================================================
def retrieval_node(state: TicketState) -> dict:
    start_time = time.time()
    metrics = state.get("latency_metrics", {})
    error_logs = state.get("errors", [])
    threshold = state.get("threshold", ESCALATION_CONFIG["log_similarity_threshold"])
    
    try:
        search_text = state.get("query", "")
        recent_ctx = state.get("recent_context", "")
        if recent_ctx:
            search_text = f"Context: {recent_ctx}\nCustomer Query: {search_text}"
            
        query_vector = embed_query(search_text)

        doc_results = pinecone_index.query(
            vector=query_vector, top_k=3,
            include_metadata=True, namespace=NAMESPACE_DOCS
        )
        support_contexts = [
            {"source": "support-docs", **match.get("metadata", {})}
            for match in doc_results.get("matches", [])
            if match.get("score", 0) >= threshold
        ]

        log_results = pinecone_index.query(
            vector=query_vector, top_k=2,
            include_metadata=True, namespace=NAMESPACE_LOGS
        )
        past_cases = [
            {
                "source"       : "action-logs",
                "score"        : round(match["score"], 4),
                "query"        : match["metadata"].get("query", ""),
                "severity"     : match["metadata"].get("severity_score", "N/A"),
                "issue_type"   : match["metadata"].get("issue_type", ""),
                "actions_taken": match["metadata"].get("actions_taken", ""),
                "resolution"   : match["metadata"].get("resolution", "")
            }
            for match in log_results.get("matches", [])
            if match.get("score", 0) >= threshold
        ]
    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        support_contexts = []
        past_cases = []
        error_logs.append(f"retrieval_error: {str(e)}")

    metrics["retrieval_node"] = round(time.time() - start_time, 2)
    return {"support_contexts": support_contexts, "past_cases": past_cases, "latency_metrics": metrics, "errors": error_logs}


def evaluation_node(state: TicketState) -> dict:
    start_time = time.time()
    metrics = state.get("latency_metrics", {})
    tokens = state.get("token_metrics", {"input": 0, "output": 0, "total": 0})
    error_logs = state.get("errors", [])
    
    dispute_threshold = ESCALATION_CONFIG["past_dispute_threshold"]
    rating_threshold  = ESCALATION_CONFIG["seller_rating_threshold"]
    value_threshold   = ESCALATION_CONFIG["high_value_threshold"]

    context_text = "\n".join([
        f"- {c.get('intent', '')} | {c.get('instruction', '')}"
        for c in state.get("support_contexts", [])
    ])
    
    past_case_text = "\n".join([
        f"- Severity {c.get('severity')} | {c.get('issue_type')} | {c.get('query', '')[:60]} -> {c.get('actions_taken', '')}"
        for c in state.get("past_cases", [])
    ]) if state.get("past_cases") else "None found."

    prompt = f"""
You are a customer support evaluation specialist for a C2C marketplace.

PREVIOUS CONVERSATION HISTORY:
{state.get('chat_history', 'None')}

CURRENT CUSTOMER QUERY: {state.get('query', '')}

RETRIEVED SIMILAR SUPPORT CASES (for context):
{context_text}

SIMILAR PAST CASES FROM HISTORY:
{past_case_text}

TRANSACTION CONTEXT:
- User type         : {state.get('user_type', 'unknown')}
- Transaction value : ${state.get('transaction_value', 0)}
- Seller rating     : {state.get('seller_rating', 'N/A')}
- Past disputes     : {state.get('past_disputes', 0)}

Severity guide:
1 = routine inquiry, no urgency
2 = minor issue, mildly inconvenienced
3 = moderate issue, noticeably frustrated
4 = serious issue, financially impacted or repeated contact
5 = critical, legal implications or threatening to leave

Additional marketplace rules (apply AFTER base severity):
- transaction_value > ${value_threshold} AND dispute -> minimum severity 4
- past_disputes > {dispute_threshold} -> add 1 to severity score
- seller_rating < {rating_threshold} AND user_type = buyer -> set trust_safety = True
- Off-platform payment OR account hacking mentioned -> severity 5 + trust_safety = True
"""

    try:
        result = eval_llm.invoke(prompt)
        eval_dict = result["parsed"].model_dump()
        
        # Extract and accumulate token usage robustly
        i_toks, o_toks, t_toks = _extract_tokens(result["raw"])
        tokens["input"] += i_toks
        tokens["output"] += o_toks
        tokens["total"] += t_toks
        
    except Exception as e:
        eval_dict = {
            "severity_score": 5, "severity_label": "system_error", "sentiment": "unknown",
            "issue_type": "ai_evaluation_failed", "repeat_issue": False, "escalate": True,
            "trust_safety": True, "reasoning": f"LLM evaluation failed. Error: {str(e)}",
            "suggested_tone": "professional"
        }
        error_logs.append(f"evaluation_error: {str(e)}")

    metrics["evaluation_node"] = round(time.time() - start_time, 2)
    return {"evaluation": eval_dict, "latency_metrics": metrics, "token_metrics": tokens, "errors": error_logs}


def human_review_node(state: TicketState) -> dict:
    start_time = time.time()
    metrics = state.get("latency_metrics", {})
    eval_result = state.get("evaluation", {})
    reason = "Trust & Safety concern detected" if eval_result.get("trust_safety") else "Automated evaluation failed"

    canned_response = (
        "Thank you for reaching out. Your case has been flagged for priority review "
        "by our specialized support team. A human agent will contact you within 2 hours. "
        "We apologize for any inconvenience caused."
    )

    updated_eval = {
        **eval_result,
        "issue_type"    : eval_result.get("issue_type", "ai_evaluation_failed"),
        "escalate"      : True,
        "severity_score": eval_result.get("severity_score", 5),
        "severity_label": eval_result.get("severity_label", "critical"),
        "reasoning"     : reason
    }
    supervisor_dec = {
        "escalated"     : True, "action"        : "assign_human", "compensation"  : "none",
        "assigned_to"   : "trust_safety_team" if eval_result.get("trust_safety") else "human_agent",
        "priority"      : "urgent", "internal_notes": "Human review required - DLQ case"
    }

    metrics["human_review_node"] = round(time.time() - start_time, 2)
    return {
        "customer_response"  : canned_response,
        "evaluation"         : updated_eval,
        "supervisor_decision": supervisor_dec,
        "latency_metrics"    : metrics,
        "token_metrics"      : state.get("token_metrics", {"input": 0, "output": 0, "total": 0})
    }


def response_node(state: TicketState) -> dict:
    start_time = time.time()
    metrics = state.get("latency_metrics", {})
    tokens = state.get("token_metrics", {"input": 0, "output": 0, "total": 0})
    error_logs = state.get("errors", [])
    
    eval_data = state.get("evaluation", {})
    context_text = "\n".join([
        f"Question: {c.get('instruction', '')}\nAnswer: {c.get('response', '')}"
        for c in state.get("support_contexts", [])
    ])

    tone_instructions = {
        "professional": "Respond clearly and professionally.",
        "empathetic"  : "Acknowledge the customer's frustration first, then provide a solution.",
        "urgent"      : "Acknowledge the severity immediately, apologise sincerely, and prioritise resolution."
    }
    tone_guide = tone_instructions.get(eval_data.get("suggested_tone"), tone_instructions["professional"])

    prompt = f"""
You are a customer support agent for a C2C marketplace.
{tone_guide}
Keep your response to 3-5 sentences. Do not mention internal severity scores.

RETRIEVED CONTEXT:
{context_text}

PREVIOUS CONVERSATION HISTORY:
{state.get('chat_history', 'None')}

CURRENT CUSTOMER QUERY: {state.get('query', '')}
CUSTOMER SENTIMENT: {eval_data.get('sentiment', 'neutral')}
YOUR RESPONSE:"""

    try:
        response = lc_llm.invoke(prompt)
        answer = response.content.strip()
        
        # Extract and accumulate token usage robustly
        i_toks, o_toks, t_toks = _extract_tokens(response)
        tokens["input"] += i_toks
        tokens["output"] += o_toks
        tokens["total"] += t_toks
        
    except Exception as e:
        answer = "Thank you for contacting us. We've received your message and our team is looking into this. We'll get back to you as soon as possible."
        error_logs.append(f"response_error: {str(e)}")

    metrics["response_node"] = round(time.time() - start_time, 2)
    return {"customer_response": answer, "latency_metrics": metrics, "token_metrics": tokens, "errors": error_logs}


def supervisor_node(state: TicketState) -> dict:
    start_time = time.time()
    metrics = state.get("latency_metrics", {})
    tokens = state.get("token_metrics", {"input": 0, "output": 0, "total": 0})
    error_logs = state.get("errors", [])
    eval_data = state.get("evaluation", {})

    if not eval_data.get("escalate"):
        metrics["supervisor_node"] = round(time.time() - start_time, 2)
        return {
            "supervisor_decision": {
                "escalated": False, "action": "none", "compensation": "none",
                "assigned_to": "none", "priority": "normal", "internal_notes": "Automated resolution."
            },
            "latency_metrics": metrics,
            "token_metrics"  : tokens    # ← pass through accumulated tokens
        }

    past_cases_text = "\n".join([
        f"- Severity {c.get('severity')} | {c.get('query', '')[:50]} -> {c.get('actions_taken', '')}"
        for c in state.get("past_cases", [])
    ]) if state.get("past_cases") else "No similar past cases found."

    trust_note = "\n⚠️ Trust & Safety flag raised — consider routing to trust_safety_team." if eval_data.get("trust_safety") else ""

    prompt = f"""
You are a senior customer support supervisor for a C2C marketplace.
A case has been escalated. Decide on the appropriate internal actions.

CASE DETAILS:
- Query        : {state.get('query', '')}
- Severity     : {eval_data.get('severity_score')}/5
- Sentiment    : {eval_data.get('sentiment')}
- Issue type   : {eval_data.get('issue_type')}
- Trust Safety : {eval_data.get('trust_safety', False)}
- Reasoning    : {eval_data.get('reasoning')}
{trust_note}

RESPONSE SENT TO CUSTOMER:
{state.get('customer_response', '')}

SIMILAR PAST CASES:
{past_cases_text}

Decision guide:
- Severity 4 + billing/dispute -> compensate + assign human_agent
- Severity 4 + delivery        -> assign human_agent + monitor
- Severity 5                   -> escalate_management + compensate + urgent
- trust_safety = True          -> assign trust_safety_team
- Repeat issue in past cases   -> escalate_rca
- Legal mentioned              -> assign legal_team
"""

    try:
        result = supervisor_llm.invoke(prompt)
        decision = result["parsed"].model_dump()
        
        # Extract and accumulate token usage robustly
        i_toks, o_toks, t_toks = _extract_tokens(result["raw"])
        tokens["input"] += i_toks
        tokens["output"] += o_toks
        tokens["total"] += t_toks
        
    except Exception as e:
        decision = {
            "escalated"     : True, "action"        : "assign_human", "compensation"  : "none",
            "assigned_to"   : "human_agent", "priority"      : "high",
            "internal_notes": f"Supervisor LLM failed. Error: {str(e)}"
        }
        error_logs.append(f"supervisor_error: {str(e)}")

    metrics["supervisor_node"] = round(time.time() - start_time, 2)
    return {"supervisor_decision": decision, "latency_metrics": metrics, "token_metrics": tokens, "errors": error_logs}


def audit_node(state: TicketState) -> dict:
    start_time = time.time()
    metrics = state.get("latency_metrics", {})
    error_logs = state.get("errors", [])
    
    ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    supervisor_decision = state.get("supervisor_decision") or {
        "escalated": False, "action": "none", "compensation": "none",
        "assigned_to": "none", "priority": "normal", "internal_notes": "Automated resolution."
    }
    eval_data = state.get("evaluation", {})

    ticket = {
        "ticket_id"        : ticket_id,
        "timestamp"        : timestamp,
        "query"            : state.get("query", ""),
        "customer_response": state.get("customer_response", ""),
        "severity_score"   : eval_data.get("severity_score", 0),
        "severity_label"   : eval_data.get("severity_label", "unknown"),
        "sentiment"        : eval_data.get("sentiment", "unknown"),
        "issue_type"       : eval_data.get("issue_type", "unknown"),
        "repeat_issue"     : str(eval_data.get("repeat_issue", False)),
        "escalated"        : str(eval_data.get("escalate", False)),
        "trust_safety"     : str(eval_data.get("trust_safety", False)),
        "reasoning"        : eval_data.get("reasoning", ""),
        "action"           : supervisor_decision.get("action", "none"),
        "compensation"     : supervisor_decision.get("compensation", "none"),
        "assigned_to"      : supervisor_decision.get("assigned_to", "none"),
        "priority"         : supervisor_decision.get("priority", "normal"),
        "internal_notes"   : supervisor_decision.get("internal_notes", ""),
        "actions_taken"    : f"{supervisor_decision.get('action','none')} | {supervisor_decision.get('compensation','none')}",
        "resolution"       : "human_review" if eval_data.get("issue_type") == "ai_evaluation_failed" else ("pending" if eval_data.get("escalate") else "resolved")
    }

    try:
        pinecone_index.upsert(
            vectors=[{"id": ticket_id, "values": embed_query(state.get("query", "")), "metadata": ticket}],
            namespace=NAMESPACE_LOGS
        )
    except Exception as e:
        error_logs.append(f"audit_error: {str(e)}")

    metrics["audit_node"] = round(time.time() - start_time, 2)

    return {
        "ticket_id"          : ticket_id,
        "supervisor_decision": supervisor_decision,
        "latency_metrics"    : metrics,
        "token_metrics"      : state.get("token_metrics", {"input": 0, "output": 0, "total": 0}),  # ← 加這行
        "errors"             : error_logs
    }
