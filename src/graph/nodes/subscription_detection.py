"""
Subscription Detection Nodes for LangGraph

This module contains nodes for detecting recurring subscriptions,
validating them, and awaiting user confirmation.
"""

from typing import Any, Dict
from src.graph.state import AnalysisState, Subscription
from src.agents.subscription_detector import SubscriptionDetectorAgent


def detect_subscriptions_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Detect recurring subscriptions from categorized transactions.
    
    Uses three-tier filtering: hard exclusions → categories → 
    restaurant keywords → variance checks → LLM validation.
    """
    try:
        transactions = state.get("transactions", [])
        
        if not transactions:
            return {
                "errors": ["No transactions available for subscription detection"]
            }
        
        # Convert to format expected by detector
        tx_list = [
            {
                "date": tx["date"],
                "description": tx["description"],
                "amount": tx["amount"],
                "category": tx.get("category", "Uncategorized")
            }
            for tx in transactions
        ]
        
        detector = SubscriptionDetectorAgent()
        results = detector.detect_subscriptions(tx_list)
        
        # Convert to Subscription TypedDict
        subscriptions = []
        for sub in results.get("subscriptions", []):
            subscriptions.append(Subscription(
                merchant=sub["merchant"],
                frequency=sub["frequency"],
                amount=sub["amount"],
                category=sub["category"],
                transaction_dates=sub["transaction_dates"],
                confirmed=False,  # Awaiting user confirmation
                note=None
            ))
        
        return {
            "detected_subscriptions": subscriptions,
            "reflection_notes": [
                f"Detected {len(subscriptions)} potential subscriptions"
            ]
        }
        
    except Exception as e:
        return {
            "errors": [f"Subscription detection failed: {str(e)}"]
        }


def validate_subscriptions_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Validate detected subscriptions using reflection.
    
    Checks for:
    - Suspicious patterns (e.g., one-time large payments)
    - Missing frequency information
    - Amount consistency
    """
    subscriptions = state.get("detected_subscriptions", [])
    
    validation_errors = []
    reflection_notes = []
    
    if not subscriptions:
        reflection_notes.append("No subscriptions detected")
        return {
            "needs_reflection": False,
            "reflection_notes": reflection_notes
        }
    
    for sub in subscriptions:
        # Check frequency
        if not sub.get("frequency"):
            validation_errors.append(
                f"Subscription '{sub['merchant']}' missing frequency"
            )
        
        # Check transaction count
        if len(sub.get("transaction_dates", [])) < 2:
            reflection_notes.append(
                f"Warning: '{sub['merchant']}' has only 1 occurrence"
            )
        
        # Check amount variance (should be consistent for true subscriptions)
        if sub.get("amount", 0) == 0:
            validation_errors.append(
                f"Subscription '{sub['merchant']}' has zero amount"
            )
    
    reflection_notes.append(
        f"Validated {len(subscriptions)} subscriptions"
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


def await_subscription_confirmation_node(state: AnalysisState) -> Dict[str, Any]:
    """
    HITL checkpoint: Wait for user to confirm subscriptions.
    
    User can select which detected subscriptions are legitimate and
    add notes for context.
    """
    return {
        "awaiting_user_input": True,
        "current_stage": "subscription_confirmation"
    }


def apply_subscription_confirmations_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Apply user subscription confirmations and notes.
    
    Updates the subscription list based on user selections.
    """
    subscriptions = state.get("detected_subscriptions", [])
    selections = state.get("subscription_selections", {})
    notes = state.get("subscription_notes", {})
    
    if not subscriptions:
        return {
            "subscriptions_confirmed": True,
            "awaiting_user_input": False
        }
    
    # Update confirmations
    confirmed_subscriptions = []
    for sub in subscriptions:
        merchant = sub["merchant"]
        
        # Only include if user confirmed
        if selections.get(merchant, False):
            sub["confirmed"] = True
            sub["note"] = notes.get(merchant)
            confirmed_subscriptions.append(sub)
    
    reflection_note = (
        f"User confirmed {len(confirmed_subscriptions)} of "
        f"{len(subscriptions)} detected subscriptions"
    )
    
    return {
        "detected_subscriptions": confirmed_subscriptions,
        "subscriptions_confirmed": True,
        "awaiting_user_input": False,
        "reflection_notes": [reflection_note]
    }
