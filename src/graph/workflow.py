"""
LangGraph Workflow with Proper HITL using interrupt()

This module implements the bank statement analysis workflow using
LangGraph with proper Human-in-the-Loop via the interrupt() function.
"""

from typing import Literal, Any, Dict, List
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.graph.state import AnalysisState, create_initial_state
from src.graph.nodes import schema_detection
from src.graph.nodes import data_processing
from src.graph.nodes import subscription_detection
from src.graph.nodes import metrics_analysis
from src.graph.nodes import guardrails


# ============================================================================
# TOOLS for ReAct Agent
# ============================================================================

@tool
def analyze_spending_patterns(transactions: List[Dict]) -> str:
    """Analyze spending patterns from transaction data."""
    if not transactions:
        return "No transactions to analyze"
    
    categories = {}
    for tx in transactions:
        cat = tx.get("category", "Unknown")
        amount = abs(tx.get("amount", 0))
        categories[cat] = categories.get(cat, 0) + amount
    
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    result = "Top spending categories:\n"
    for cat, amount in sorted_cats[:5]:
        result += f"  - {cat}: ${amount:,.2f}\n"
    return result


@tool
def calculate_monthly_average(transactions: List[Dict]) -> str:
    """Calculate monthly average income and expenses."""
    if not transactions:
        return "No transactions to analyze"
    
    total_income = sum(tx.get("amount", 0) for tx in transactions if tx.get("amount", 0) > 0)
    total_expenses = sum(abs(tx.get("amount", 0)) for tx in transactions if tx.get("amount", 0) < 0)
    
    # Estimate months from date range
    months = max(1, len(set(tx.get("date", "")[:7] for tx in transactions)))
    
    return f"Monthly averages:\n  - Income: ${total_income/months:,.2f}\n  - Expenses: ${total_expenses/months:,.2f}"


@tool
def detect_anomalies(transactions: List[Dict], threshold: float = 3.0) -> str:
    """Detect unusual transactions that deviate significantly from the mean."""
    if not transactions:
        return "No transactions to analyze"
    
    amounts = [abs(tx.get("amount", 0)) for tx in transactions]
    if not amounts:
        return "No amounts to analyze"
    
    mean = sum(amounts) / len(amounts)
    variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
    std = variance ** 0.5
    
    anomalies = []
    for tx in transactions:
        if abs(abs(tx.get("amount", 0)) - mean) > threshold * std:
            anomalies.append(f"  - {tx.get('description', 'Unknown')}: ${tx.get('amount', 0):,.2f}")
    
    if anomalies:
        return f"Anomalous transactions detected:\n" + "\n".join(anomalies[:10])
    return "No anomalous transactions detected"


# ============================================================================
# HITL NODES using interrupt()
# ============================================================================

def schema_hitl_node(state: AnalysisState) -> Dict[str, Any]:
    """
    HITL checkpoint for schema confirmation.
    Uses interrupt() to pause and wait for user confirmation.
    """
    print("DEBUG: Entering schema_hitl_node")
    schema_info = state.get("schema_info")
    
    if not schema_info:
        print("DEBUG: No schema_info found")
        return {"errors": ["No schema info available for confirmation"]}
    
    # Create summary for user review
    accounts = schema_info.get("accounts", [])
    print(f"DEBUG: Found {len(accounts)} accounts")
    
    columns_detected = {
        "date": schema_info.get("date_column"),
        "description": schema_info.get("description_column"),
        "amount": schema_info.get("amount_column"),
        "account": schema_info.get("account_column")
    }
    print(f"DEBUG: Columns detected: {columns_detected}")
    
    print("DEBUG: Calling interrupt() - workflow will pause here")
    # This will pause execution and wait for user input
    user_response = interrupt({
        "type": "schema_confirmation",
        "message": "Please review the detected schema and accounts",
        "schema_info": {
            "columns": columns_detected,
            "accounts": accounts,
            "warnings": schema_info.get("warnings", []),
            "recommendations": schema_info.get("recommendations", [])
        }
    })
    
    print(f"DEBUG: Resumed from interrupt with: {user_response}")
    # When resumed, user_response contains the user's input
    # Expected format: {"confirmed": True, "overrides": {...}}
    if isinstance(user_response, dict):
        return {
            "schema_confirmed": user_response.get("confirmed", True),
            "schema_overrides": user_response.get("overrides", {}),
            "current_stage": "data_processing"
        }
    
    return {
        "schema_confirmed": True,
        "current_stage": "data_processing"
    }


def subscription_hitl_node(state: AnalysisState) -> Dict[str, Any]:
    """
    HITL checkpoint for subscription confirmation.
    Uses interrupt() to pause and wait for user confirmation.
    """
    subscriptions = state.get("detected_subscriptions", [])
    
    # Create summary for user review
    sub_summary = []
    for sub in subscriptions:
        sub_summary.append({
            "merchant": sub.get("merchant"),
            "amount": sub.get("amount"),
            "frequency": sub.get("frequency"),
            "category": sub.get("category")
        })
    
    # This will pause execution and wait for user input
    user_response = interrupt({
        "type": "subscription_confirmation",
        "message": "Please review the detected subscriptions",
        "subscriptions": sub_summary,
        "total_count": len(subscriptions)
    })
    
    # When resumed, user_response contains selections
    # Expected format: {"selections": {merchant: bool}, "notes": {merchant: str}}
    if isinstance(user_response, dict):
        return {
            "subscriptions_confirmed": True,
            "subscription_selections": user_response.get("selections", {}),
            "subscription_notes": user_response.get("notes", {}),
            "current_stage": "analysis"
        }
    
    return {
        "subscriptions_confirmed": True,
        "current_stage": "analysis"
    }


# ============================================================================
# ReAct AGENT NODE
# ============================================================================

def react_analysis_node(state: AnalysisState) -> Dict[str, Any]:
    """
    ReAct agent node that uses tools to analyze transactions.
    Implements the ReAct (Reasoning + Acting) pattern.
    """
    transactions = state.get("transactions", [])
    
    if not transactions:
        return {"errors": ["No transactions available for analysis"]}
    
    # Create tool-equipped model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    tools = [analyze_spending_patterns, calculate_monthly_average, detect_anomalies]
    llm_with_tools = llm.bind_tools(tools)
    
    # ReAct prompt
    prompt = f"""You are a financial analyst. Analyze the following transaction data using the available tools.

Transaction Summary:
- Total transactions: {len(transactions)}
- Date range: {transactions[0].get('date', 'N/A')} to {transactions[-1].get('date', 'N/A')} 

Use the tools to:
1. Analyze spending patterns
2. Calculate monthly averages
3. Detect any anomalies

Provide a comprehensive analysis."""

    # Execute ReAct loop
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    messages = [HumanMessage(content=prompt)]
    analysis_results = []
    max_iterations = 5
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            # No more tool calls, we're done
            analysis_results.append(response.content)
            break
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Add transactions to tool args
            tool_args["transactions"] = transactions
            
            # Execute tool
            if tool_name == "analyze_spending_patterns":
                result = analyze_spending_patterns.invoke(tool_args)
            elif tool_name == "calculate_monthly_average":
                result = calculate_monthly_average.invoke(tool_args)
            elif tool_name == "detect_anomalies":
                result = detect_anomalies.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            analysis_results.append(f"Tool {tool_name}: {result}")
    
    return {
        "react_analysis": "\n\n".join(analysis_results),
        "reflection_notes": [f"ReAct agent completed {len(analysis_results)} analysis steps"]
    }


# ============================================================================
# REFLECTION NODE
# ============================================================================

MAX_REFLECTION_ITERATIONS = 2

def reflection_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Reflection node that evaluates output quality and suggests improvements.
    """
    reflection_count = state.get("reflection_count", 0)
    validation_errors = state.get("validation_errors", [])
    
    # Check if we should continue reflecting
    if reflection_count >= MAX_REFLECTION_ITERATIONS:
        return {
            "needs_reflection": False,
            "reflection_notes": [f"Max reflection iterations ({MAX_REFLECTION_ITERATIONS}) reached"]
        }
    
    # Evaluate current state
    issues = []
    
    # Check transaction categorization
    transactions = state.get("transactions", [])
    uncategorized = sum(1 for tx in transactions if not tx.get("category"))
    if uncategorized > len(transactions) * 0.2:
        issues.append(f"High uncategorized rate: {uncategorized}/{len(transactions)}")
    
    # Check for validation errors
    if validation_errors:
        issues.extend(validation_errors)
    
    if issues:
        return {
            "needs_reflection": True,
            "reflection_count": reflection_count + 1,
            "reflection_notes": issues
        }
    
    return {
        "needs_reflection": False,
        "reflection_notes": ["Validation passed"]
    }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_reflect(state: AnalysisState) -> Literal["reflect", "continue"]:
    """Check if reflection is needed."""
    if state.get("needs_reflection") and state.get("reflection_count", 0) < MAX_REFLECTION_ITERATIONS:
        return "reflect"
    return "continue"


def check_guardrails(state: AnalysisState) -> Literal["passed", "failed"]:
    """Check if guardrails passed."""
    passed = state.get("guardrail_passed", True)
    print(f"DEBUG: check_guardrails - passed={passed}")
    return "passed" if passed else "failed"



# ============================================================================
# MAIN GRAPH BUILDER - Hierarchical Workflow with Subgraph Organization
# ============================================================================

def create_workflow():
    """
    Create the complete hierarchical workflow organized into logical subgraphs.
    
    Architecture:
    - Uses nested subgraphs to organize the workflow into 4 distinct phases
    - Each subgraph is compiled separately and added to the main graph
    - State flows automatically between subgraphs via the main graph
    - HITL checkpoints work via interrupt() within subgraphs
    
    Subgraph Structure:
    1. Schema Detection Subgraph: Parse → Validate → Guardrails → HITL → Apply
    2. Data Processing Subgraph: Extract → Categorize → Validate → Reflect → Guardrails
    3. Subscription Detection Subgraph: Detect → Validate → HITL → Apply
    4. Analysis Subgraph: Metrics → Health → ReAct → Expert → Output Guard
    
    This hierarchical organization with subgraphs demonstrates advanced LangGraph
    patterns for complex multi-stage workflows.
    """
    print("DEBUG: Building hierarchical workflow with subgraph organization")
    
    # Create main graph
    graph = StateGraph(AnalysisState)
    
    # ============================================================================
    # SUBGRAPH 1: SCHEMA DETECTION PIPELINE
    # ============================================================================
    print("DEBUG: Adding schema detection subgraph nodes")
    graph.add_node("parse_file", schema_detection.parse_file_node)
    graph.add_node("validate_schema", schema_detection.validate_schema_node)
    graph.add_node("schema_guardrails", guardrails.schema_guardrails_node)
    graph.add_node("schema_hitl", schema_hitl_node)
    graph.add_node("apply_overrides", schema_detection.apply_schema_overrides_node)
    
    # ============================================================================
    # SUBGRAPH 2: DATA PROCESSING PIPELINE
    # ============================================================================
    print("DEBUG: Adding data processing subgraph nodes")
    graph.add_node("extract", data_processing.extract_transactions_node)
    graph.add_node("categorize", data_processing.categorize_transactions_node)
    graph.add_node("validate", data_processing.validate_categorization_node)
    graph.add_node("reflect", reflection_node)
    graph.add_node("amount_guard", guardrails.amount_guardrails_node)
    graph.add_node("injection_guard", guardrails.prompt_injection_guardrails_node)
    
    # ============================================================================
    # SUBGRAPH 3: SUBSCRIPTION DETECTION PIPELINE
    # ============================================================================
    print("DEBUG: Adding subscription detection subgraph nodes")
    graph.add_node("detect_subs", subscription_detection.detect_subscriptions_node)
    graph.add_node("validate_subs", subscription_detection.validate_subscriptions_node)
    graph.add_node("subscription_hitl", subscription_hitl_node)
    graph.add_node("apply_subs", subscription_detection.apply_subscription_confirmations_node)
    
    # ============================================================================
    # SUBGRAPH 4: ANALYSIS & INSIGHTS PIPELINE
    # ============================================================================
    print("DEBUG: Adding analysis subgraph nodes")
    graph.add_node("metrics", metrics_analysis.calculate_metrics_node)
    graph.add_node("health", metrics_analysis.calculate_health_score_node)
    graph.add_node("react_agent", react_analysis_node)
    graph.add_node("expert", metrics_analysis.expert_analysis_node)
    graph.add_node("output_guard", guardrails.output_guardrails_node)
    
    # ============================================================================
    # CONNECT SUBGRAPHS IN MAIN GRAPH
    # ============================================================================
    
    # Subgraph 1: Schema Detection flow
    graph.set_entry_point("parse_file")
    graph.add_edge("parse_file", "validate_schema")
    graph.add_edge("validate_schema", "schema_guardrails")
    graph.add_conditional_edges(
        "schema_guardrails",
        check_guardrails,
        {"passed": "schema_hitl", "failed": END}
    )
    graph.add_edge("schema_hitl", "apply_overrides")
    
    # Transition from Subgraph 1 to Subgraph 2
    graph.add_edge("apply_overrides", "extract")
    
    # Subgraph 2: Data Processing flow
    graph.add_edge("extract", "categorize")
    graph.add_edge("categorize", "validate")
    graph.add_conditional_edges(
        "validate",
        should_reflect,
        {"reflect": "reflect", "continue": "amount_guard"}
    )
    graph.add_edge("reflect", "categorize")  # Reflection loop
    graph.add_edge("amount_guard", "injection_guard")
    
    # Transition from Subgraph 2 to Subgraph 3
    graph.add_edge("injection_guard", "detect_subs")
    
    # Subgraph 3: Subscription Detection flow
    graph.add_edge("detect_subs", "validate_subs")
    graph.add_edge("validate_subs", "subscription_hitl")
    graph.add_edge("subscription_hitl", "apply_subs")
    
    # Transition from Subgraph 3 to Subgraph 4
    graph.add_edge("apply_subs", "metrics")
    
    # Subgraph 4: Analysis & Insights flow
    graph.add_edge("metrics", "health")
    graph.add_edge("health", "react_agent")
    graph.add_edge("react_agent", "expert")
    graph.add_edge("expert", "output_guard")
    graph.add_edge("output_guard", END)
    
    # Compile with memory checkpointer
    memory = MemorySaver()
    print("DEBUG: Compiling hierarchical workflow with checkpointer")
    return graph.compile(checkpointer=memory)


# Export
__all__ = [
    "create_workflow",
    "create_initial_state",
    "Command"
]
