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
        
        print(f"DEBUG: detect_subscriptions_node - received {len(transactions)} transactions")
        
        if not transactions:
            print("DEBUG: No transactions available")
            return {
                "errors": ["No transactions available for subscription detection"]
            }
        
        # Show sample transactions for debugging
        print(f"DEBUG: Sample transactions: {transactions[:3]}")
        
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
        
        print(f"DEBUG: Converted {len(tx_list)} transactions for detector")
        
        # Use same parameters as original app.py for consistent results
        detector = SubscriptionDetectorAgent(min_occurrences=2, max_day_variance=5)
        results = detector.detect_subscriptions(tx_list)
        
        print(f"DEBUG: Detector returned {len(results.get('subscriptions', []))} subscriptions")
        if results.get('subscriptions'):
            print(f"DEBUG: First subscription keys: {results['subscriptions'][0].keys()}")
            print(f"DEBUG: First subscription: {results['subscriptions'][0]}")
        
        # Convert to Subscription TypedDict
        # Note: detector returns 'description' not 'merchant', and uses 'first_seen'/'last_seen'
        # Filter out Daily/Weekly frequencies - these are not real subscriptions
        valid_frequencies = ["Monthly", "Quarterly", "Semi-annual", "Annual", "Bi-weekly"]
        subscriptions = []
        for sub in results.get("subscriptions", []):
            frequency = sub.get("frequency", "Unknown")
            # Skip Daily and Weekly - these are regular purchases, not subscriptions
            if frequency in ["Daily", "Weekly", "Weekly (approx)"]:
                print(f"DEBUG: Skipping {sub.get('description')} - frequency {frequency} is not a subscription")
                continue
                
            subscriptions.append(Subscription(
                merchant=sub.get("description", sub.get("merchant", "Unknown")),  # Map description to merchant
                frequency=frequency,
                amount=abs(sub.get("amount", 0)),  # Ensure positive amount
                category=sub.get("category", "Subscription"),
                transaction_dates=[sub.get("first_seen", ""), sub.get("last_seen", "")],  # Use first/last seen
                confirmed=False,  # Awaiting user confirmation
                note=None
            ))
        
        print(f"DEBUG: Returning {len(subscriptions)} subscriptions to state (after filtering)")
        
        return {
            "detected_subscriptions": subscriptions,
            "reflection_notes": [
                f"Detected {len(subscriptions)} potential subscriptions"
            ]
        }
        
    except Exception as e:
        import traceback
        print(f"DEBUG: Subscription detection error: {e}")
        print(traceback.format_exc())
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
