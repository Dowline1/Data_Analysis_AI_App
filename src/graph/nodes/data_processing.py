"""
Data Processing Nodes for LangGraph

This module contains nodes for extracting transactions, categorizing them,
and validating the processed data with reflection loops.
"""

from typing import Any, Dict, List
import pandas as pd
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
        raw_data = state.get("raw_data", {})
        
        print(f"DEBUG: extract_transactions_node - file_path: {file_path}")
        print(f"DEBUG: extract_transactions_node - schema_info type: {type(schema_info)}")
        print(f"DEBUG: extract_transactions_node - raw_data keys: {raw_data.keys() if raw_data else 'None'}")
        
        # Get DataFrame from raw_data (saved during parse_file_node)
        df_dict = raw_data.get("dataframe") if raw_data else None
        file_type = raw_data.get("file_type", "xlsx") if raw_data else "xlsx"
        
        if df_dict:
            df = pd.DataFrame.from_dict(df_dict)
            print(f"DEBUG: extract_transactions_node - DataFrame reconstructed with {len(df)} rows")
        else:
            # Fallback: re-parse the file
            from src.tools.file_parser import FileParser
            parsed = FileParser.parse_file(file_path)
            df = parsed['dataframe']
            file_type = parsed['file_type']
            print(f"DEBUG: extract_transactions_node - Re-parsed file with {len(df)} rows")
        
        # Use SchemaMapperAgent to extract transactions
        mapper = SchemaMapperAgent()
        extracted_transactions = mapper.extract_from_dataframe(df, file_type, file_path)
        
        print(f"DEBUG: extract_transactions_node - extracted {len(extracted_transactions)} raw transactions")
        
        # Helper to convert date to string
        def date_to_string(date_val):
            if date_val is None:
                return ""
            if isinstance(date_val, str):
                return date_val
            if hasattr(date_val, 'strftime'):  # datetime or Timestamp
                return date_val.strftime('%Y-%m-%d')
            return str(date_val)
        
        # Convert to Transaction TypedDict
        transactions = []
        for tx in extracted_transactions:
            transactions.append(Transaction(
                date=date_to_string(tx.get("date", "")),
                description=str(tx.get("description", "")),
                amount=float(tx.get("amount", 0.0)),
                account_type=str(tx.get("account_type", "unknown")),
                category=tx.get("category"),  # May be set by mapper or None
                transaction_type=tx.get("transaction_type"),
                balance=tx.get("balance")
            ))
        
        date_range = None
        if transactions:
            dates = [tx["date"] for tx in transactions if tx.get("date")]
            if dates:
                date_range = (min(dates), max(dates))
        
        print(f"DEBUG: extract_transactions_node - returning {len(transactions)} transactions")
        
        return {
            "transactions": transactions,
            "total_transactions": len(transactions),
            "date_range": date_range,
            "reflection_notes": [f"Extracted {len(transactions)} transactions"],
            "reflection_count": 0,
            "validation_errors": []
        }
        
    except Exception as e:
        import traceback
        print(f"DEBUG: extract_transactions_node - ERROR: {str(e)}")
        print(f"DEBUG: {traceback.format_exc()}")
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
        
        print(f"DEBUG: categorize_transactions_node - received {len(transactions)} transactions")
        
        if not transactions:
            print("DEBUG: categorize_transactions_node - NO TRANSACTIONS!")
            return {
                "errors": ["No transactions to categorize"]
            }
        
        categorizer = TransactionCategorizerAgent()
        
        # Convert to dict format expected by categorizer
        tx_dicts = [dict(tx) for tx in transactions]
        
        # Categorize all transactions at once
        categorized_tx_dicts = categorizer.categorize_transactions(tx_dicts)
        
        # Convert back to Transaction TypedDict
        categorized_transactions = []
        for tx in categorized_tx_dicts:
            categorized_transactions.append(Transaction(
                date=tx["date"],
                description=tx["description"],
                amount=tx["amount"],
                account_type=tx["account_type"],
                category=tx.get("category", "Other"),
                transaction_type=tx.get("transaction_type"),
                balance=tx.get("balance")
            ))
        
        print(f"DEBUG: categorize_transactions_node - categorized {len(categorized_transactions)} transactions")
        
        return {
            "transactions": categorized_transactions,
            "categorization_complete": True,
            "reflection_notes": [
                f"Categorized {len(categorized_transactions)} transactions"
            ]
        }
        
    except Exception as e:
        import traceback
        print(f"DEBUG: categorize_transactions_node - ERROR: {e}")
        print(traceback.format_exc())
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
