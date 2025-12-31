"""
LangGraph State Schema for Bank Statement Analysis System

This module defines the typed state that flows through the LangGraph workflow.
The state is shared across all nodes and subgraphs, enabling data flow and
persistence across the multi-stage analysis pipeline.
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from operator import add


class AccountInfo(TypedDict):
    """Information about a detected account."""
    account_name: str
    account_type: str  # credit_card, checking, current, savings
    transaction_count: int
    sample_descriptions: List[str]


class SchemaInfo(TypedDict):
    """Schema detection results from initial file parsing."""
    columns_found: List[str]
    date_column: Optional[str]
    description_column: Optional[str]
    amount_column: Optional[str]
    account_column: Optional[str]
    balance_column: Optional[str]
    accounts: List[AccountInfo]
    warnings: List[str]
    recommendations: List[str]


class Transaction(TypedDict):
    """Individual transaction record."""
    date: str
    description: str
    amount: float
    account_type: str
    category: Optional[str]
    transaction_type: Optional[str]  # debit, credit
    balance: Optional[float]


class Subscription(TypedDict):
    """Detected recurring subscription."""
    merchant: str
    frequency: str
    amount: float
    category: str
    transaction_dates: List[str]
    confirmed: bool
    note: Optional[str]


class AccountMetrics(TypedDict):
    """Financial metrics for a single account."""
    account_type: str
    total_income: float
    total_expenses: float
    net_cash_flow: float
    
    # Credit card specific
    total_charges: Optional[float]
    total_payments: Optional[float]
    payment_ratio: Optional[float]
    net_balance: Optional[float]
    
    # Checking/savings specific
    savings_rate: Optional[float]
    avg_monthly_income: Optional[float]
    avg_monthly_expenses: Optional[float]


class HealthScore(TypedDict):
    """Financial health scoring."""
    overall_score: int  # 0-100
    checking_score: int
    credit_card_score: int
    income_ratio_score: int
    savings_rate_score: int
    payment_ratio_score: int
    balance_health_score: int
    assessment: str  # EXCELLENT, GOOD, FAIR, NEEDS_ATTENTION


class ExpertInsights(TypedDict):
    """Expert analysis insights."""
    spending_patterns: str
    income_analysis: str
    credit_card_health: str
    subscription_impact: str
    recommendations: List[str]
    warnings: List[str]


class AnalysisState(TypedDict):
    """
    Main state for the LangGraph workflow.
    
    This state flows through all nodes and tracks the complete analysis pipeline
    from file upload through schema detection, data processing, subscription
    analysis, and expert review.
    """
    
    # Input
    file_path: str
    file_content: Optional[bytes]
    
    # Schema Detection Stage
    schema_info: Optional[SchemaInfo]
    schema_confirmed: bool
    schema_overrides: Optional[Dict[str, str]]  # account_name -> account_type
    
    # Data Processing Stage
    transactions: Annotated[List[Transaction], add]
    raw_data: Optional[Dict[str, Any]]
    categorization_complete: bool
    
    # Subscription Detection Stage
    detected_subscriptions: List[Subscription]
    subscriptions_confirmed: bool
    subscription_selections: Optional[Dict[str, bool]]  # merchant -> confirmed
    subscription_notes: Optional[Dict[str, str]]  # merchant -> note
    
    # Metrics Calculation Stage
    account_metrics: List[AccountMetrics]
    health_score: Optional[HealthScore]
    
    # Expert Analysis Stage
    expert_insights: Optional[ExpertInsights]
    expert_report: Optional[str]
    
    # Reflection & Validation
    validation_errors: Annotated[List[str], add]
    reflection_notes: Annotated[List[str], add]
    needs_reflection: bool
    reflection_count: int
    
    # Guardrails
    guardrail_passed: bool
    guardrail_violations: Annotated[List[str], add]
    
    # Error Handling
    errors: Annotated[List[str], add]
    current_stage: str  # schema_detection, data_processing, subscription_analysis, metrics, expert_review
    
    # HITL Checkpoints
    awaiting_user_input: bool
    user_feedback: Optional[str]
    
    # Metadata
    analysis_timestamp: Optional[str]
    total_transactions: int
    date_range: Optional[tuple[str, str]]


def create_initial_state(file_path: str) -> AnalysisState:
    """
    Create an initial state for a new analysis.
    
    Args:
        file_path: Path to the bank statement file
        
    Returns:
        Initial AnalysisState with defaults
    """
    return AnalysisState(
        # Input
        file_path=file_path,
        file_content=None,
        
        # Schema Detection
        schema_info=None,
        schema_confirmed=False,
        schema_overrides=None,
        
        # Data Processing
        transactions=[],
        raw_data=None,
        categorization_complete=False,
        
        # Subscriptions
        detected_subscriptions=[],
        subscriptions_confirmed=False,
        subscription_selections=None,
        subscription_notes=None,
        
        # Metrics
        account_metrics=[],
        health_score=None,
        
        # Expert Analysis
        expert_insights=None,
        expert_report=None,
        
        # Reflection
        validation_errors=[],
        reflection_notes=[],
        needs_reflection=False,
        reflection_count=0,
        
        # Guardrails
        guardrail_passed=True,
        guardrail_violations=[],
        
        # Error Handling
        errors=[],
        current_stage="schema_detection",
        
        # HITL
        awaiting_user_input=False,
        user_feedback=None,
        
        # Metadata
        analysis_timestamp=None,
        total_transactions=0,
        date_range=None,
    )
