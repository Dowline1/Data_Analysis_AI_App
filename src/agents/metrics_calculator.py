"""
Metrics Calculator Agent - CodeAct Pattern
Executes Python code (pandas) to calculate financial metrics and statistics.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.logger import setup_logger
from src.state.app_state import Transaction

logger = setup_logger(__name__)


class MetricsCalculatorAgent:
    """
    CodeAct agent that executes Python code to calculate financial metrics.
    Uses pandas for efficient computation on transaction data.
    """
    
    def __init__(self):
        pass
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def calculate_all_metrics(self, transactions: List[Transaction]) -> Dict:
        """
        Execute Python code to calculate all financial metrics.
        
        CodeAct Pattern: Executes actual pandas/numpy operations.
        
        Args:
            transactions: List of transaction dicts
            
        Returns:
            Dict with all calculated metrics
        """
        logger.info(f"Calculating metrics for {len(transactions)} transactions using pandas")
        
        if not transactions:
            logger.warning("No transactions to calculate metrics")
            return self._empty_metrics()
        
        # Convert to DataFrame for efficient computation (CodeAct)
        df = pd.DataFrame(transactions)
        
        # Execute metric calculations
        metrics = {
            'basic': self._calculate_basic_metrics(df),
            'spending': self._calculate_spending_metrics(df),
            'income': self._calculate_income_metrics(df),
            'trends': self._calculate_trends(df),
            'health': self._calculate_financial_health(df)
        }
        
        # Convert numpy types to native Python types for serialization
        metrics = self._convert_numpy_types(metrics)
        
        logger.info("Metrics calculation complete")
        
        return metrics
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict:
        """Execute Python code for basic transaction metrics."""
        logger.debug("Calculating basic metrics")
        
        return {
            'total_transactions': int(len(df)),
            'total_income': float(df[df['amount'] > 0]['amount'].sum()) if len(df[df['amount'] > 0]) > 0 else 0.0,
            'total_expenses': float(abs(df[df['amount'] < 0]['amount'].sum())) if len(df[df['amount'] < 0]) > 0 else 0.0,
            'net_balance': float(df['amount'].sum()),
            'average_transaction': float(df['amount'].mean()),
            'largest_expense': float(df[df['amount'] < 0]['amount'].min()) if len(df[df['amount'] < 0]) > 0 else 0.0,
            'largest_income': float(df[df['amount'] > 0]['amount'].max()) if len(df[df['amount'] > 0]) > 0 else 0.0,
        }
    
    def _calculate_spending_metrics(self, df: pd.DataFrame) -> Dict:
        """Execute Python code for spending analysis."""
        logger.debug("Calculating spending metrics")
        
        expenses = df[df['amount'] < 0].copy()
        
        if len(expenses) == 0:
            return {
                'total_spending': 0.0,
                'average_spend': 0.0,
                'spending_by_category': {},
                'top_expenses': []
            }
        
        # Execute pandas operations
        expenses['amount_abs'] = expenses['amount'].abs()
        
        # Category breakdown
        if 'category' in expenses.columns:
            spending_by_category = expenses.groupby('category')['amount_abs'].sum().to_dict()
        else:
            spending_by_category = {}
        
        # Top 5 expenses
        top_expenses = expenses.nlargest(5, 'amount_abs')[['date', 'description', 'amount', 'category']].to_dict('records')
        
        return {
            'total_spending': float(expenses['amount_abs'].sum()),
            'average_spend': float(expenses['amount_abs'].mean()),
            'spending_by_category': {k: float(v) for k, v in spending_by_category.items()},
            'top_expenses': [
                {
                    'date': str(exp['date']),
                    'description': exp['description'],
                    'amount': float(exp['amount']),
                    'category': exp.get('category', 'Other')
                }
                for exp in top_expenses
            ]
        }
    
    def _calculate_income_metrics(self, df: pd.DataFrame) -> Dict:
        """Execute Python code for income analysis."""
        logger.debug("Calculating income metrics")
        
        income = df[df['amount'] > 0].copy()
        
        if len(income) == 0:
            return {
                'total_income': 0.0,
                'average_income': 0.0,
                'income_sources': 0,
                'top_income': []
            }
        
        # Top 5 income transactions
        top_income = income.nlargest(5, 'amount')[['date', 'description', 'amount']].to_dict('records')
        
        return {
            'total_income': float(income['amount'].sum()),
            'average_income': float(income['amount'].mean()),
            'income_sources': int(income['description'].nunique()),
            'top_income': [
                {
                    'date': str(inc['date']),
                    'description': inc['description'],
                    'amount': float(inc['amount'])
                }
                for inc in top_income
            ]
        }
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict:
        """Execute Python code for trend analysis."""
        logger.debug("Calculating trends")
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date for daily trends
        daily_spending = df[df['amount'] < 0].groupby(df['date'].dt.date)['amount'].sum().abs()
        daily_income = df[df['amount'] > 0].groupby(df['date'].dt.date)['amount'].sum()
        
        # Calculate weekly averages
        df['week'] = df['date'].dt.isocalendar().week
        weekly_spending = df[df['amount'] < 0].groupby('week')['amount'].sum().abs()
        
        return {
            'daily_average_spending': float(daily_spending.mean()) if len(daily_spending) > 0 else 0.0,
            'daily_average_income': float(daily_income.mean()) if len(daily_income) > 0 else 0.0,
            'weekly_average_spending': float(weekly_spending.mean()) if len(weekly_spending) > 0 else 0.0,
            'spending_volatility': float(daily_spending.std()) if len(daily_spending) > 1 else 0.0,
            'most_active_day': str(daily_spending.idxmax()) if len(daily_spending) > 0 else None,
            'spending_trend': 'increasing' if len(daily_spending) > 1 and daily_spending.iloc[-1] > daily_spending.iloc[0] else 'stable'
        }
    
    def _calculate_financial_health(self, df: pd.DataFrame) -> Dict:
        """
        Execute Python code to calculate financial health score.
        
        Score factors:
        - Income vs Expenses ratio (50%)
        - Spending consistency (25%)
        - Savings rate (25%)
        """
        logger.debug("Calculating financial health score")
        
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        
        if total_income == 0:
            return {
                'health_score': 0,
                'rating': 'Poor',
                'factors': {
                    'income_expense_ratio': 0,
                    'savings_rate': 0,
                    'spending_consistency': 0
                }
            }
        
        # Factor 1: Income vs Expenses (50 points max)
        income_expense_ratio = min((total_income / total_expenses) * 50, 50) if total_expenses > 0 else 50
        
        # Factor 2: Savings rate (25 points max)
        net_savings = total_income - total_expenses
        savings_rate = (net_savings / total_income) * 25 if total_income > 0 else 0
        savings_rate = max(0, min(savings_rate, 25))  # Clamp between 0-25
        
        # Factor 3: Spending consistency (25 points max)
        expenses = df[df['amount'] < 0]['amount'].abs()
        consistency_score = 25 - min((expenses.std() / expenses.mean()) * 10, 25) if len(expenses) > 1 else 25
        
        # Total score
        health_score = int(income_expense_ratio + savings_rate + consistency_score)
        
        # Rating
        if health_score >= 80:
            rating = 'Excellent'
        elif health_score >= 60:
            rating = 'Good'
        elif health_score >= 40:
            rating = 'Fair'
        else:
            rating = 'Poor'
        
        return {
            'health_score': health_score,
            'rating': rating,
            'factors': {
                'income_expense_ratio': round(income_expense_ratio, 2),
                'savings_rate': round(savings_rate, 2),
                'spending_consistency': round(consistency_score, 2)
            },
            'net_savings': float(net_savings),
            'savings_percentage': round((net_savings / total_income) * 100, 2) if total_income > 0 else 0
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure when no transactions."""
        return {
            'basic': {
                'total_transactions': 0,
                'total_income': 0.0,
                'total_expenses': 0.0,
                'net_balance': 0.0,
                'average_transaction': 0.0,
                'largest_expense': 0.0,
                'largest_income': 0.0
            },
            'spending': {
                'total_spending': 0.0,
                'average_spend': 0.0,
                'spending_by_category': {},
                'top_expenses': []
            },
            'income': {
                'total_income': 0.0,
                'average_income': 0.0,
                'income_sources': 0,
                'top_income': []
            },
            'trends': {
                'daily_average_spending': 0.0,
                'daily_average_income': 0.0,
                'weekly_average_spending': 0.0,
                'spending_volatility': 0.0,
                'most_active_day': None,
                'spending_trend': 'stable'
            },
            'health': {
                'health_score': 0,
                'rating': 'N/A',
                'factors': {
                    'income_expense_ratio': 0,
                    'savings_rate': 0,
                    'spending_consistency': 0
                },
                'net_savings': 0.0,
                'savings_percentage': 0.0
            }
        }
