"""
Main LangGraph Workflow for Bank Statement Analysis

This module constructs the complete StateGraph with subgraphs,
reflection loops, guardrails, and HITL checkpoints.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AnalysisState, create_initial_state
from src.graph.nodes import schema_detection
from src.graph.nodes import data_processing
from src.graph.nodes import subscription_detection
from src.graph.nodes import metrics_analysis
from src.graph.nodes import guardrails


def should_await_user_input(state: AnalysisState) -> Literal["await_user", "continue"]:
    """Conditional edge: Check if we need user input."""
    return "await_user" if state.get("awaiting_user_input") else "continue"


def should_reflect(state: AnalysisState) -> Literal["reflect", "continue"]:
    """Conditional edge: Check if reflection loop is needed."""
    return "reflect" if state.get("needs_reflection") else "continue"


def check_guardrails(state: AnalysisState) -> Literal["failed", "passed"]:
    """Conditional edge: Check if guardrails passed."""
    return "passed" if state.get("guardrail_passed", True) else "failed"


def build_schema_detection_subgraph() -> StateGraph:
    """
    Build the schema detection subgraph.
    
    Flow: parse_file → validate_schema → guardrails → [await_confirmation] → apply_overrides
    """
    subgraph = StateGraph(AnalysisState)
    
    # Add nodes
    subgraph.add_node("parse_file", schema_detection.parse_file_node)
    subgraph.add_node("validate_schema", schema_detection.validate_schema_node)
    subgraph.add_node("schema_guardrails", guardrails.schema_guardrails_node)
    subgraph.add_node("await_confirmation", schema_detection.await_schema_confirmation_node)
    subgraph.add_node("apply_overrides", schema_detection.apply_schema_overrides_node)
    
    # Set entry point
    subgraph.set_entry_point("parse_file")
    
    # Add edges
    subgraph.add_edge("parse_file", "validate_schema")
    subgraph.add_edge("validate_schema", "schema_guardrails")
    
    # Conditional: guardrails check
    subgraph.add_conditional_edges(
        "schema_guardrails",
        check_guardrails,
        {
            "passed": "await_confirmation",
            "failed": END
        }
    )
    
    # HITL checkpoint
    subgraph.add_edge("await_confirmation", "apply_overrides")
    subgraph.add_edge("apply_overrides", END)
    
    return subgraph


def build_data_processing_subgraph() -> StateGraph:
    """
    Build the data processing subgraph with reflection loop.
    
    Flow: extract → categorize → validate → [reflect if needed] → guardrails
    """
    subgraph = StateGraph(AnalysisState)
    
    # Add nodes
    subgraph.add_node("extract_transactions", data_processing.extract_transactions_node)
    subgraph.add_node("categorize", data_processing.categorize_transactions_node)
    subgraph.add_node("validate_categorization", data_processing.validate_categorization_node)
    subgraph.add_node("reflection_loop", data_processing.reflection_loop_node)
    subgraph.add_node("amount_guardrails", guardrails.amount_guardrails_node)
    subgraph.add_node("injection_guardrails", guardrails.prompt_injection_guardrails_node)
    
    # Set entry point
    subgraph.set_entry_point("extract_transactions")
    
    # Add edges
    subgraph.add_edge("extract_transactions", "categorize")
    subgraph.add_edge("categorize", "validate_categorization")
    
    # Conditional: reflection loop
    subgraph.add_conditional_edges(
        "validate_categorization",
        should_reflect,
        {
            "reflect": "reflection_loop",
            "continue": "amount_guardrails"
        }
    )
    
    # Reflection loops back to categorize
    subgraph.add_edge("reflection_loop", "categorize")
    
    # Guardrails sequence
    subgraph.add_edge("amount_guardrails", "injection_guardrails")
    
    # Final guardrails check
    subgraph.add_conditional_edges(
        "injection_guardrails",
        check_guardrails,
        {
            "passed": END,
            "failed": END
        }
    )
    
    return subgraph


def build_subscription_subgraph() -> StateGraph:
    """
    Build the subscription detection subgraph with HITL.
    
    Flow: detect → validate → [await_confirmation] → apply_confirmations
    """
    subgraph = StateGraph(AnalysisState)
    
    # Add nodes
    subgraph.add_node("detect_subscriptions", subscription_detection.detect_subscriptions_node)
    subgraph.add_node("validate_subscriptions", subscription_detection.validate_subscriptions_node)
    subgraph.add_node("await_confirmation", subscription_detection.await_subscription_confirmation_node)
    subgraph.add_node("apply_confirmations", subscription_detection.apply_subscription_confirmations_node)
    
    # Set entry point
    subgraph.set_entry_point("detect_subscriptions")
    
    # Add edges
    subgraph.add_edge("detect_subscriptions", "validate_subscriptions")
    
    # Conditional: reflection check (optional warnings)
    subgraph.add_conditional_edges(
        "validate_subscriptions",
        should_reflect,
        {
            "reflect": "await_confirmation",  # Show warnings to user
            "continue": "await_confirmation"  # Proceed to confirmation anyway
        }
    )
    
    # HITL checkpoint
    subgraph.add_edge("await_confirmation", "apply_confirmations")
    subgraph.add_edge("apply_confirmations", END)
    
    return subgraph


def build_analysis_subgraph() -> StateGraph:
    """
    Build the metrics and expert analysis subgraph.
    
    Flow: calculate_metrics → health_score → expert_analysis → validate → guardrails
    """
    subgraph = StateGraph(AnalysisState)
    
    # Add nodes
    subgraph.add_node("calculate_metrics", metrics_analysis.calculate_metrics_node)
    subgraph.add_node("calculate_health", metrics_analysis.calculate_health_score_node)
    subgraph.add_node("expert_analysis", metrics_analysis.expert_analysis_node)
    subgraph.add_node("validate_analysis", metrics_analysis.validate_analysis_node)
    subgraph.add_node("output_guardrails", guardrails.output_guardrails_node)
    
    # Set entry point
    subgraph.set_entry_point("calculate_metrics")
    
    # Add edges
    subgraph.add_edge("calculate_metrics", "calculate_health")
    subgraph.add_edge("calculate_health", "expert_analysis")
    subgraph.add_edge("expert_analysis", "validate_analysis")
    
    # Conditional: final validation
    subgraph.add_conditional_edges(
        "validate_analysis",
        should_reflect,
        {
            "reflect": END,  # Exit with warnings
            "continue": "output_guardrails"
        }
    )
    
    # Final guardrails
    subgraph.add_edge("output_guardrails", END)
    
    return subgraph


def build_main_graph() -> StateGraph:
    """
    Build the main LangGraph workflow.
    
    Orchestrates the four subgraphs with HITL checkpoints and error handling.
    """
    # Build subgraphs
    schema_graph = build_schema_detection_subgraph()
    processing_graph = build_data_processing_subgraph()
    subscription_graph = build_subscription_subgraph()
    analysis_graph = build_analysis_subgraph()
    
    # Create main graph
    main_graph = StateGraph(AnalysisState)
    
    # Add subgraph nodes
    main_graph.add_node("schema_detection", schema_graph.compile())
    main_graph.add_node("data_processing", processing_graph.compile())
    main_graph.add_node("subscription_detection", subscription_graph.compile())
    main_graph.add_node("expert_review", analysis_graph.compile())
    
    # Set entry point
    main_graph.set_entry_point("schema_detection")
    
    # Connect subgraphs sequentially
    main_graph.add_edge("schema_detection", "data_processing")
    main_graph.add_edge("data_processing", "subscription_detection")
    main_graph.add_edge("subscription_detection", "expert_review")
    main_graph.add_edge("expert_review", END)
    
    return main_graph


def create_analysis_workflow():
    """
    Create the complete compiled workflow with checkpointing.
    
    Returns:
        Compiled LangGraph application with memory saver for HITL.
    """
    graph = build_main_graph()
    
    # Add checkpointer for HITL support
    memory = MemorySaver()
    
    # Compile with checkpointing
    app = graph.compile(checkpointer=memory)
    
    return app


# Export for use in Streamlit app
__all__ = [
    "create_analysis_workflow",
    "create_initial_state",
    "AnalysisState"
]
