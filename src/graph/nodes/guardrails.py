"""
Guardrails Nodes for LangGraph

This module contains guardrail nodes that validate data safety,
check for anomalies, and prevent malicious inputs.
"""

from typing import Any, Dict
from datetime import datetime
from src.graph.state import AnalysisState


def schema_guardrails_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Guardrails for schema detection stage.
    
    Validates:
    - File path is safe (no path traversal)
    - Column names are reasonable (no injection attempts)
    - Account types are from allowed list
    """
    print("DEBUG: Entering schema_guardrails_node")
    violations = []
    
    # Check file path safety
    file_path = state.get("file_path", "")
    if ".." in file_path or file_path.startswith("/"):
        violations.append("Unsafe file path detected")
    
    # Check schema info if available
    schema_info = state.get("schema_info")
    if schema_info:
        # Validate column names (alphanumeric + underscore only)
        columns = schema_info.get("columns_found", [])
        for col in columns:
            if not col.replace("_", "").replace(" ", "").isalnum():
                violations.append(f"Suspicious column name: {col}")
        
        # Validate account types - allow common account type patterns
        # The LLM may detect various credit card types (platinum_card, silver_card, etc.)
        allowed_types = {
            "credit_card", "checking", "current", "savings", "unknown",
            "platinum_card", "gold_card", "silver_card", "rewards_card",
            "debit", "investment", "money_market", "certificate", "loan"
        }
        for acc in schema_info.get("accounts", []):
            acc_type = acc["account_type"].lower().replace(" ", "_")
            # Accept any card type or account type ending in common suffixes
            is_valid = (
                acc_type in allowed_types or
                acc_type.endswith("_card") or
                acc_type.endswith("_account") or
                "credit" in acc_type or
                "checking" in acc_type or
                "savings" in acc_type
            )
            if not is_valid:
                violations.append(
                    f"Invalid account type: {acc['account_type']}"
                )
    
    if violations:
        print(f"DEBUG: schema_guardrails violations: {violations}")
        return {
            "guardrail_passed": False,
            "guardrail_violations": violations
        }
    
    print("DEBUG: schema_guardrails passed, returning guardrail_passed=True")
    return {
        "guardrail_passed": True
    }


def amount_guardrails_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Guardrails for transaction amounts.
    
    Validates:
    - Amounts are within reasonable ranges
    - No extreme outliers that suggest data corruption
    - Date values are valid and recent
    """
    violations = []
    transactions = state.get("transactions", [])
    
    if not transactions:
        return {"guardrail_passed": True}
    
    # Check amount ranges
    amounts = [abs(tx["amount"]) for tx in transactions]
    max_amount = max(amounts) if amounts else 0
    
    # Flag suspiciously large amounts (> $100k)
    if max_amount > 100000:
        violations.append(
            f"Suspiciously large transaction: ${max_amount:,.2f}"
        )
    
    # Check for too many zero amounts
    zero_count = sum(1 for amt in amounts if amt == 0)
    if zero_count > len(amounts) * 0.1:
        violations.append(
            f"Too many zero-amount transactions: {zero_count}"
        )
    
    # Validate dates
    current_year = datetime.now().year
    for tx in transactions[:100]:  # Sample first 100
        try:
            tx_date_raw = tx.get("date")
            
            # Handle different date types
            if tx_date_raw is None:
                continue
            elif isinstance(tx_date_raw, datetime):
                tx_date = tx_date_raw
            elif isinstance(tx_date_raw, str):
                tx_date = datetime.fromisoformat(tx_date_raw.replace('Z', '+00:00'))
            elif hasattr(tx_date_raw, 'to_pydatetime'):  # pandas Timestamp
                tx_date = tx_date_raw.to_pydatetime()
            else:
                # Try to convert to string first
                tx_date = datetime.fromisoformat(str(tx_date_raw))
            
            if tx_date.year < 2000 or tx_date.year > current_year:
                violations.append(
                    f"Invalid transaction date: {tx_date_raw}"
                )
                break
        except (ValueError, AttributeError, TypeError) as e:
            violations.append(f"Malformed date: {tx.get('date')} ({type(tx.get('date')).__name__})")
            break
    
    if violations:
        return {
            "guardrail_passed": False,
            "guardrail_violations": violations
        }
    
    return {
        "guardrail_passed": True
    }


def prompt_injection_guardrails_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Guardrails to prevent prompt injection attacks.
    
    Validates:
    - Transaction descriptions don't contain malicious prompts
    - User feedback doesn't contain injection attempts
    - No attempts to manipulate LLM behavior
    """
    violations = []
    
    # Check transaction descriptions (sample)
    transactions = state.get("transactions", [])
    suspicious_patterns = [
        "ignore previous",
        "ignore all",
        "new instructions",
        "system:",
        "admin:",
        "<script>",
        "DROP TABLE",
        "'; --"
    ]
    
    for tx in transactions[:50]:  # Sample first 50
        desc = tx.get("description", "").lower()
        for pattern in suspicious_patterns:
            if pattern in desc:
                violations.append(
                    f"Suspicious pattern in description: {pattern}"
                )
                break
    
    # Check user feedback
    user_feedback = state.get("user_feedback", "")
    if user_feedback:
        feedback_lower = user_feedback.lower()
        for pattern in suspicious_patterns:
            if pattern in feedback_lower:
                violations.append(
                    f"Suspicious pattern in user feedback: {pattern}"
                )
                break
    
    if violations:
        return {
            "guardrail_passed": False,
            "guardrail_violations": violations
        }
    
    return {
        "guardrail_passed": True
    }


def output_guardrails_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Guardrails for final output validation.
    
    Validates:
    - Expert report is reasonable length
    - No sensitive data leakage
    - Recommendations are actionable
    """
    violations = []
    
    # Check expert report
    expert_report = state.get("expert_report", "")
    if expert_report:
        # Check length (should be substantial but not excessive)
        if len(expert_report) < 100:
            violations.append("Expert report too short")
        elif len(expert_report) > 50000:
            violations.append("Expert report excessively long")
        
        # Check for PII patterns (basic check)
        import re
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', expert_report):
            violations.append("Potential SSN detected in report")
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', expert_report):
            violations.append("Potential credit card number in report")
    
    # Check recommendations
    expert_insights = state.get("expert_insights")
    if expert_insights:
        recommendations = expert_insights.get("recommendations", [])
        if not recommendations:
            violations.append("No recommendations provided")
        elif len(recommendations) > 20:
            violations.append("Too many recommendations (overwhelming)")
    
    if violations:
        return {
            "guardrail_passed": False,
            "guardrail_violations": violations
        }
    
    return {
        "guardrail_passed": True
    }
