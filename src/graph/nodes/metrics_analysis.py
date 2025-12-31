"""
Metrics and Analysis Nodes for LangGraph

This module contains nodes for calculating financial metrics,
health scoring, and expert analysis.
"""

from typing import Any, Dict
from src.graph.state import AnalysisState, AccountMetrics, HealthScore, ExpertInsights
from src.agents.metrics_calculator import MetricsCalculatorAgent
from src.agents.expert_insights_agent import ExpertInsightsAgent


def calculate_metrics_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Calculate per-account financial metrics.
    
    Separates logic for credit cards (charges/payments/ratios) vs
    checking/savings (income/expenses/savings rate).
    """
    try:
        transactions = state.get("transactions", [])
        
        if not transactions:
            return {
                "errors": ["No transactions available for metrics calculation"]
            }
        
        calculator = MetricsCalculatorAgent()
        results = calculator.calculate_metrics(transactions)
        
        # Convert to AccountMetrics TypedDict
        account_metrics = []
        for acc_type, metrics in results.get("per_account_metrics", {}).items():
            account_metrics.append(AccountMetrics(
                account_type=acc_type,
                total_income=metrics.get("total_income", 0),
                total_expenses=metrics.get("total_expenses", 0),
                net_cash_flow=metrics.get("net_cash_flow", 0),
                total_charges=metrics.get("total_charges"),
                total_payments=metrics.get("total_payments"),
                payment_ratio=metrics.get("payment_ratio"),
                net_balance=metrics.get("net_balance"),
                savings_rate=metrics.get("savings_rate"),
                avg_monthly_income=metrics.get("avg_monthly_income"),
                avg_monthly_expenses=metrics.get("avg_monthly_expenses")
            ))
        
        return {
            "account_metrics": account_metrics,
            "reflection_notes": [
                f"Calculated metrics for {len(account_metrics)} accounts"
            ]
        }
        
    except Exception as e:
        return {
            "errors": [f"Metrics calculation failed: {str(e)}"]
        }


def calculate_health_score_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Calculate account-aware financial health score.
    
    Checking/savings: 60 points (income ratio 40 + savings 20)
    Credit cards: 40 points (payment ratio 30 + balance health 10)
    
    Recognizes CC overpayments (negative balance) as positive indicator.
    """
    try:
        account_metrics = state.get("account_metrics", [])
        transactions = state.get("transactions", [])
        
        if not account_metrics:
            return {
                "errors": ["No account metrics available for health scoring"]
            }
        
        calculator = MetricsCalculatorAgent()
        
        # Build metrics dict for calculator
        metrics_dict = {
            "per_account_metrics": {}
        }
        for acc in account_metrics:
            metrics_dict["per_account_metrics"][acc["account_type"]] = {
                "total_income": acc["total_income"],
                "total_expenses": acc["total_expenses"],
                "net_cash_flow": acc["net_cash_flow"],
                "total_charges": acc.get("total_charges"),
                "total_payments": acc.get("total_payments"),
                "payment_ratio": acc.get("payment_ratio"),
                "net_balance": acc.get("net_balance"),
                "savings_rate": acc.get("savings_rate"),
            }
        
        health_result = calculator._calculate_financial_health(
            transactions=transactions,
            metrics=metrics_dict
        )
        
        health_score = HealthScore(
            overall_score=health_result["overall_score"],
            checking_score=health_result["checking_score"],
            credit_card_score=health_result["credit_card_score"],
            income_ratio_score=health_result["income_ratio_score"],
            savings_rate_score=health_result["savings_rate_score"],
            payment_ratio_score=health_result["payment_ratio_score"],
            balance_health_score=health_result["balance_health_score"],
            assessment=health_result["assessment"]
        )
        
        return {
            "health_score": health_score,
            "reflection_notes": [
                f"Health score: {health_score['overall_score']}/100 - {health_score['assessment']}"
            ]
        }
        
    except Exception as e:
        return {
            "errors": [f"Health score calculation failed: {str(e)}"]
        }


def expert_analysis_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Generate expert insights using full ledger review.
    
    ReAct node that analyzes spending patterns, income, credit card health,
    subscription impact, and generates personalized recommendations.
    """
    try:
        transactions = state.get("transactions", [])
        account_metrics = state.get("account_metrics", [])
        subscriptions = state.get("detected_subscriptions", [])
        health_score = state.get("health_score")
        
        if not transactions:
            return {
                "errors": ["No transactions available for expert analysis"]
            }
        
        agent = ExpertInsightsAgent()
        
        # Build context for expert
        context = {
            "transactions": transactions,
            "account_metrics": account_metrics,
            "subscriptions": subscriptions,
            "health_score": health_score,
            "date_range": state.get("date_range")
        }
        
        result = agent.analyze_transactions(context)
        
        expert_insights = ExpertInsights(
            spending_patterns=result.get("spending_patterns", ""),
            income_analysis=result.get("income_analysis", ""),
            credit_card_health=result.get("credit_card_health", ""),
            subscription_impact=result.get("subscription_impact", ""),
            recommendations=result.get("recommendations", []),
            warnings=result.get("warnings", [])
        )
        
        return {
            "expert_insights": expert_insights,
            "expert_report": result.get("full_report"),
            "reflection_notes": [
                "Expert analysis complete with personalized recommendations"
            ]
        }
        
    except Exception as e:
        return {
            "errors": [f"Expert analysis failed: {str(e)}"]
        }


def validate_analysis_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Validate final analysis for completeness and quality.
    
    Reflection node that ensures all stages completed successfully.
    """
    validation_errors = []
    reflection_notes = []
    
    # Check each stage
    if not state.get("schema_confirmed"):
        validation_errors.append("Schema not confirmed")
    
    if not state.get("transactions"):
        validation_errors.append("No transactions extracted")
    
    if not state.get("categorization_complete"):
        validation_errors.append("Categorization incomplete")
    
    if not state.get("account_metrics"):
        validation_errors.append("No metrics calculated")
    
    if not state.get("health_score"):
        validation_errors.append("No health score calculated")
    
    if not state.get("expert_report"):
        validation_errors.append("No expert report generated")
    
    # Check for accumulated errors
    if state.get("errors"):
        validation_errors.extend(state["errors"])
    
    reflection_notes.append(
        f"Analysis validation: {len(validation_errors)} errors found"
    )
    
    if validation_errors:
        return {
            "validation_errors": validation_errors,
            "needs_reflection": True,
            "reflection_notes": reflection_notes
        }
    
    return {
        "needs_reflection": False,
        "reflection_notes": [
            "Analysis complete - all stages validated successfully"
        ]
    }
