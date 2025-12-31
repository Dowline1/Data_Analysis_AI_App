"""
Schema Detection Nodes for LangGraph

This module contains nodes for the schema detection subgraph, including
file parsing, column identification, account detection, and validation.
"""

from typing import Any, Dict
from datetime import datetime
from pathlib import Path
from src.graph.state import AnalysisState, SchemaInfo, AccountInfo
from src.agents.schema_preview_agent import SchemaPreviewAgent
from src.tools.file_parser import FileParser


def parse_file_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Parse the uploaded file and detect basic structure.
    
    This is the entry point for the schema detection subgraph.
    """
    try:
        file_path = state["file_path"]
        
        # Parse file using FileParser first
        parsed = FileParser.parse_file(file_path)
        df = parsed['dataframe']
        file_type = parsed['file_type']
        filename = Path(file_path).name
        
        # Generate schema preview
        agent = SchemaPreviewAgent()
        preview_result = agent.generate_preview(df, file_type, filename)
        
        # Map preview result to our schema structure
        detected_cols = preview_result.get("detected_columns", {})
        account_preview = preview_result.get("account_preview", [])
        
        # Extract schema information
        schema_info = SchemaInfo(
            columns_found=list(df.columns),
            date_column=detected_cols.get("date_column"),
            description_column=detected_cols.get("description_column"),
            amount_column=detected_cols.get("amount_column"),
            account_column=detected_cols.get("account_name_column"),
            balance_column=detected_cols.get("balance_column"),
            accounts=[
                AccountInfo(
                    account_name=acc["account_name"],
                    account_type=acc["account_type"],
                    transaction_count=acc["transaction_count"],
                    sample_descriptions=acc.get("sample_transactions", [])
                )
                for acc in account_preview
            ],
            warnings=preview_result.get("warnings", []),
            recommendations=preview_result.get("recommendations", [])
        )
        
        return {
            "schema_info": schema_info,
            "raw_data": {"dataframe": df.to_dict(), "file_type": file_type, "columns": list(df.columns)},
            "current_stage": "schema_detection",
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        return {
            "errors": [f"Schema detection failed: {str(e)}\n{traceback.format_exc()}"],
            "current_stage": "schema_detection"
        }


def validate_schema_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Validate the detected schema for completeness and correctness.
    
    This is a reflection node that checks if schema detection was successful
    and all required columns were found.
    """
    schema_info = state.get("schema_info")
    
    if not schema_info:
        return {
            "validation_errors": ["No schema information available"],
            "needs_reflection": True
        }
    
    validation_errors = []
    
    # Check required columns
    required_columns = ["date_column", "description_column", "amount_column"]
    for col in required_columns:
        if not schema_info.get(col):
            validation_errors.append(f"Missing required column: {col}")
    
    # Check accounts detected
    if not schema_info.get("accounts"):
        validation_errors.append("No accounts detected in file")
    
    # Check for unknown account types
    unknown_accounts = [
        acc["account_name"] 
        for acc in schema_info.get("accounts", [])
        if acc["account_type"] == "unknown"
    ]
    if unknown_accounts:
        validation_errors.append(
            f"Unknown account types detected: {', '.join(unknown_accounts)}"
        )
    
    if validation_errors:
        return {
            "validation_errors": validation_errors,
            "needs_reflection": True,
            "reflection_notes": [
                "Schema validation found issues that may need user confirmation"
            ]
        }
    
    return {
        "needs_reflection": False,
        "reflection_notes": ["Schema validation passed"]
    }


def await_schema_confirmation_node(state: AnalysisState) -> Dict[str, Any]:
    """
    HITL checkpoint: Wait for user to confirm or override schema detection.
    
    This node sets a flag to interrupt the graph and wait for user input.
    """
    return {
        "awaiting_user_input": True,
        "current_stage": "schema_confirmation"
    }


def apply_schema_overrides_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Apply user-provided schema overrides after confirmation.
    
    This node processes any account type corrections from the user.
    """
    schema_info = state.get("schema_info")
    overrides = state.get("schema_overrides", {})
    
    if not schema_info or not overrides:
        return {
            "schema_confirmed": True,
            "awaiting_user_input": False
        }
    
    # Apply overrides to accounts
    updated_accounts = []
    for acc in schema_info.get("accounts", []):
        if acc["account_name"] in overrides:
            acc["account_type"] = overrides[acc["account_name"]]
        updated_accounts.append(acc)
    
    schema_info["accounts"] = updated_accounts
    
    return {
        "schema_info": schema_info,
        "schema_confirmed": True,
        "awaiting_user_input": False,
        "reflection_notes": [f"Applied {len(overrides)} schema overrides"]
    }
