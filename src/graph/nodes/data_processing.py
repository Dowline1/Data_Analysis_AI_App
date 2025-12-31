"""
Data Processing Nodes for LangGraph

This module contains nodes for extracting transactions, categorizing them,
and validating the processed data with reflection loops.
"""

from typing import Any, Dict, List
from src.graph.state import AnalysisState, Transaction
from src.agents.schema_mapper import SchemaMapperAgent
from src.agents.transaction_categorizer import TransactionCategorizerAgent


def extract_transactions_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Extract and normalize transactions from the raw file.
    
    Uses the confirmed schema to parse transactions with account-aware
    sign conventions.
    """
    try:
        schema_info = state["schema_info"]
        file_path = state["file_path"]
        
        # Build column mapping from schema
        column_mapping = {
            "date": schema_info["date_column"],
            "description": schema_info["description_column"],
            "amount": schema_info["amount_column"],
        }
        
        if schema_info.get("account_column"):
            column_mapping["account"] = schema_info["account_column"]
        if schema_info.get("balance_column"):
            column_mapping["balance"] = schema_info["balance_column"]
        
        # Extract transactions
        mapper = SchemaMapperAgent()
        result = mapper.extract_transactions(
            file_path=file_path,
            column_mapping=column_mapping
        )
        
        # Convert to Transaction TypedDict
        transactions = []
        for tx in result.get("transactions", []):
            transactions.append(Transaction(
                date=tx["date"],
                description=tx["description"],
                amount=tx["amount"],
                account_type=tx["account_type"],
                category=None,  # Will be set by categorization
                transaction_type=tx.get("transaction_type"),
                balance=tx.get("balance")
            ))
        
        date_range = None
        if transactions:
            dates = [tx["date"] for tx in transactions]
            date_range = (min(dates), max(dates))
        
        return {
            "transactions": transactions,
            "total_transactions": len(transactions),
            "date_range": date_range,
            "raw_data": result,
            "reflection_notes": [f"Extracted {len(transactions)} transactions"]
        }
        
    except Exception as e:
        return {
            "errors": [f"Transaction extraction failed: {str(e)}"]
        }


def categorize_transactions_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Categorize all transactions using hybrid keyword + LLM approach.
    
    This is a CodeAct node that uses tools (keyword matching, LLM) to
    classify transactions.
    """
    try:
        transactions = state.get("transactions", [])
        
        if not transactions:
            return {
                "errors": ["No transactions to categorize"]
            }
        
        categorizer = TransactionCategorizerAgent()
        
        # Categorize each transaction
        categorized_transactions = []
        for tx in transactions:
            category = categorizer.categorize(
                description=tx["description"],
                amount=tx["amount"],
                account_type=tx["account_type"]
            )
            
            tx["category"] = category
            categorized_transactions.append(tx)
        
        return {
            "transactions": categorized_transactions,
            "categorization_complete": True,
            "reflection_notes": [
                f"Categorized {len(categorized_transactions)} transactions"
            ]
        }
        
    except Exception as e:
        return {
            "errors": [f"Categorization failed: {str(e)}"]
        }


def validate_categorization_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Validate categorization quality using reflection.
    
    Checks for:
    - Uncategorized transactions
    - Suspicious category assignments
    - Category distribution
    """
    transactions = state.get("transactions", [])
    
    if not transactions:
        return {
            "validation_errors": ["No transactions to validate"],
            "needs_reflection": True
        }
    
    validation_errors = []
    reflection_notes = []
    
    # Check for uncategorized
    uncategorized = [tx for tx in transactions if not tx.get("category")]
    if uncategorized:
        validation_errors.append(
            f"{len(uncategorized)} transactions lack categories"
        )
    
    # Check category distribution
    categories = {}
    for tx in transactions:
        cat = tx.get("category", "Uncategorized")
        categories[cat] = categories.get(cat, 0) + 1
    
    reflection_notes.append(
        f"Category distribution: {len(categories)} unique categories"
    )
    
    # Check for suspicious patterns
    if categories.get("Uncategorized", 0) > len(transactions) * 0.1:
        validation_errors.append(
            "More than 10% of transactions are uncategorized"
        )
    
    if validation_errors:
        return {
            "validation_errors": validation_errors,
            "needs_reflection": True,
            "reflection_notes": reflection_notes
        }
    
    return {
        "needs_reflection": False,
        "reflection_notes": reflection_notes
    }


def reflection_loop_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Reflection loop to improve categorization quality.
    
    Analyzes validation errors and attempts corrections.
    Max 2 reflection iterations to prevent infinite loops.
    """
    reflection_count = state.get("reflection_count", 0)
    validation_errors = state.get("validation_errors", [])
    
    if reflection_count >= 2:
        return {
            "reflection_count": reflection_count,
            "needs_reflection": False,
            "reflection_notes": [
                f"Max reflections reached. Proceeding with {len(validation_errors)} warnings."
            ]
        }
    
    # Analyze errors and provide reflection notes
    reflection_notes = [
        f"Reflection {reflection_count + 1}: Analyzing categorization issues"
    ]
    
    for error in validation_errors:
        if "uncategorized" in error.lower():
            reflection_notes.append(
                "Consider using more aggressive LLM categorization for uncategorized items"
            )
        elif "suspicious" in error.lower():
            reflection_notes.append(
                "Review keyword matching patterns for common merchants"
            )
    
    return {
        "reflection_count": reflection_count + 1,
        "needs_reflection": False,  # Will be re-evaluated after re-categorization
        "reflection_notes": reflection_notes
    }
