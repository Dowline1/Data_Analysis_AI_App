"""
State definitions for the bank account analyzer.
This file defines the structure of data that flows through our agents.
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from datetime import datetime
import operator


class Transaction(TypedDict):
    """Represents a single bank transaction."""
    date: datetime
    description: str
    amount: float
    category: Optional[str]


class AppState(TypedDict):
    """
    Main state that gets passed between all agents in the graph.
    Using Annotated with operator.add for lists means new items get appended instead of replaced.
    """
    # Input
    file_path: str
    file_type: str  # 'csv' or 'xlsx'
    
    # Data extraction phase
    raw_data: Optional[str]  # Raw extracted text/data from file
    extracted_columns: Optional[List[str]]  # Column names found in the file
    
    # Data processing phase
    transactions: List[Transaction]  # Standardized transaction data
    total_transactions: Optional[int]
    data_quality_score: Optional[float]  # 0-1 score of data quality
    
    # Human validation
    human_approved: Optional[bool]
    human_feedback: Optional[str]
    validation_summary: Optional[Dict]  # Summary stats for human review
    
    # Analysis results
    total_spend: Optional[float]
    previous_month_spend: Optional[float]
    top_categories: Optional[List[Dict]]  # [{'category': 'Food', 'amount': 500}, ...]
    subscriptions: Optional[List[Dict]]  # Recurring transactions
    financial_health_score: Optional[float]  # 1-10 score
    
    # Insights and recommendations
    insights: Annotated[List[str], operator.add]  # AI-generated insights
    recommendations: Annotated[List[str], operator.add]  # Financial advice
    
    # Error handling and reflection
    errors: Annotated[List[str], operator.add]  # Any errors encountered
    reflection_notes: Annotated[List[str], operator.add]  # Self-correction notes
    retry_count: int  # Track retries for reflection loops
    
    # Chat interaction
    user_query: Optional[str]  # Current user question
    chat_response: Optional[str]  # AI response to user
    
    # Metadata
    processing_stage: str  # Current stage in the pipeline
    timestamp: datetime


class DataIngestionState(TypedDict):
    """
    State for the data ingestion subgraph.
    Subset of AppState focused on file parsing and validation.
    """
    file_path: str
    file_type: str
    raw_data: Optional[str]
    extracted_columns: Optional[List[str]]
    transactions: List[Transaction]
    data_quality_score: Optional[float]
    human_approved: Optional[bool]
    human_feedback: Optional[str]
    validation_summary: Optional[Dict]
    errors: Annotated[List[str], operator.add]
    reflection_notes: Annotated[List[str], operator.add]
    retry_count: int


class AnalysisState(TypedDict):
    """
    State for the analysis subgraph.
    Focused on deriving insights from validated transactions.
    """
    transactions: List[Transaction]
    total_spend: Optional[float]
    previous_month_spend: Optional[float]
    top_categories: Optional[List[Dict]]
    subscriptions: Optional[List[Dict]]
    subscription_total_cost: Optional[float]
    financial_health_score: Optional[float]
    financial_health_rating: Optional[str]
    financial_health_details: Optional[Dict]
    
    # Metrics from calculate_metrics node
    basic_metrics: Optional[Dict]
    spending_metrics: Optional[Dict]
    income_metrics: Optional[Dict]
    trend_metrics: Optional[Dict]
    
    # Category summary from categorize_transactions
    category_summary: Optional[Dict]
    
    # Alerts
    alerts: Annotated[List[str], operator.add]
    
    insights: Annotated[List[str], operator.add]
    recommendations: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    reflection_notes: Annotated[List[str], operator.add]
    analysis_complete: Optional[bool]
