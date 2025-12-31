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
            Dict with all calculated metrics including per-account breakdowns
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
        
        # Add per-account breakdowns if account_type exists
        if 'account_type' in df.columns and df['account_type'].notna().any():
            logger.info("Calculating per-account metrics")
            metrics['by_account'] = self._calculate_per_account_metrics(df)
        else:
            metrics['by_account'] = {}
        
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
        
        Score factors (account-type aware):
        - For checking/savings: Income vs Expenses ratio (50%)
        - Credit card management: Payment ratio and balance (30%)
        - Spending consistency (20%)
        
        This properly handles:
        - Credit card overpayments (negative balance = good)
        - Multiple account types with different conventions
        """
        logger.debug("Calculating financial health score")
        
        # Check if transactions have account type information for smarter analysis
        has_accounts = 'account_type' in df.columns and df['account_type'].notna().any()
        
        if has_accounts:
            # Calculate health based on account types
            checking_df = df[df['account_type'].isin(['checking', 'current', 'savings'])]
            cc_df = df[df['account_type'] == 'credit_card']
            
            # Checking/Savings health (60% weight)
            checking_score = 0
            if not checking_df.empty:
                income = checking_df[checking_df['amount'] > 0]['amount'].sum()
                expenses = abs(checking_df[checking_df['amount'] < 0]['amount'].sum())
                
                if income > 0:
                    # Income vs Expense ratio (40 points)
                    income_ratio = min((income / expenses) * 40, 40) if expenses > 0 else 40
                    
                    # Savings rate (20 points)
                    net_savings = income - expenses
                    savings_rate = (net_savings / income) * 20
                    savings_rate = max(0, min(savings_rate, 20))
                    
                    checking_score = income_ratio + savings_rate
            
            # Credit Card health (40% weight)
            cc_score = 0
            if not cc_df.empty:
                total_charges = cc_df[cc_df['amount'] > 0]['amount'].sum()
                total_payments = abs(cc_df[cc_df['amount'] < 0]['amount'].sum())
                net_balance = cc_df['amount'].sum()
                
                if total_charges > 0:
                    # Payment coverage ratio (30 points)
                    payment_ratio = min((total_payments / total_charges), 1.5) * 30  # Cap at 1.5 (overpayment is good)
                    
                    # Balance health (10 points)
                    # Negative balance = overpayment = excellent
                    # Low positive balance = good
                    # High positive balance = poor
                    if net_balance <= 0:
                        balance_score = 10  # Overpaid = excellent
                    elif net_balance < total_charges * 0.2:
                        balance_score = 7  # Low balance = good
                    elif net_balance < total_charges * 0.5:
                        balance_score = 4  # Medium balance = fair
                    else:
                        balance_score = 0  # High unpaid balance = poor
                    
                    cc_score = payment_ratio + balance_score
                elif total_payments > 0 and total_charges == 0:
                    # Paying off existing debt with no new charges = excellent
                    cc_score = 40
            
            # Combine scores
            health_score = int(checking_score + cc_score)
            
            # Analyze spending consistency (informational only, not scored)
            all_expenses = df[df['amount'] < 0]['amount'].abs()
            consistency_note = ""
            if len(all_expenses) > 1:
                cv = all_expenses.std() / all_expenses.mean()  # Coefficient of variation
                if cv < 0.5:
                    consistency_note = "Very consistent spending"
                elif cv < 1.0:
                    consistency_note = "Moderately consistent spending"
                else:
                    consistency_note = "Variable spending patterns"
        else:
            # Legacy calculation for files without account types
            total_income = df[df['amount'] > 0]['amount'].sum()
            total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
            
            if total_income == 0:
                return {
                    'health_score': 0,
                    'rating': 'Poor',
                    'factors': {},
                    'net_savings': 0,
                    'savings_percentage': 0
                }
            
            # Simple income vs expense ratio
            income_expense_ratio = min((total_income / total_expenses) * 50, 50) if total_expenses > 0 else 50
            net_savings = total_income - total_expenses
            savings_rate = (net_savings / total_income) * 50
            savings_rate = max(0, min(savings_rate, 50))
            
            health_score = int(income_expense_ratio + savings_rate)
            consistency_note = ""
        
        # Rating
        if health_score >= 80:
            rating = 'Excellent'
        elif health_score >= 60:
            rating = 'Good'
        elif health_score >= 40:
            rating = 'Fair'
        else:
            rating = 'Poor'
        
        # Calculate overall net change
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        net_savings = total_income - total_expenses
        
        return {
            'health_score': health_score,
            'rating': rating,
            'factors': {
                'account_aware': has_accounts,
                'consistency': consistency_note
            },
            'net_savings': float(net_savings),
            'savings_percentage': round((net_savings / total_income) * 100, 2) if total_income > 0 else 0
        }
    
    def _calculate_per_account_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate metrics for each account type separately.
        
        Args:
            df: DataFrame with transactions including account_type column
            
        Returns:
            Dict with metrics for each account type
        """
        logger.debug("Calculating per-account metrics")
        
        account_metrics = {}
        
        # Get unique account types
        account_types = df['account_type'].dropna().unique()
        
        for account_type in account_types:
            if not account_type or account_type == 'unknown':
                continue
                
            account_df = df[df['account_type'] == account_type]
            logger.debug(f"Processing {len(account_df)} transactions for {account_type}")
            
            # Calculate basic metrics for this account
            total_transactions = len(account_df)
            total_inflows = float(account_df[account_df['amount'] > 0]['amount'].sum()) if len(account_df[account_df['amount'] > 0]) > 0 else 0.0
            total_outflows = float(abs(account_df[account_df['amount'] < 0]['amount'].sum())) if len(account_df[account_df['amount'] < 0]) > 0 else 0.0
            net_change = float(account_df['amount'].sum())
            
            # For credit cards, interpret differently
            if account_type == 'credit_card':
                # For credit cards:
                # - Positive amounts = purchases/charges (debt increasing)
                # - Negative amounts = payments (debt decreasing)
                total_charges = total_inflows  # Purchases
                total_payments = total_outflows  # Payments
                current_balance = net_change  # Net debt
                payment_ratio = (total_payments / total_charges * 100) if total_charges > 0 else 0.0
                
                account_metrics[account_type] = {
                    'account_type': account_type,
                    'total_transactions': total_transactions,
                    'total_charges': total_charges,
                    'total_payments': total_payments,
                    'net_balance': current_balance,
                    'payment_ratio': payment_ratio,
                    'interpretation': 'Positive balance = outstanding debt'
                }
            else:
                # For checking/savings:
                # - Positive = deposits/income
                # - Negative = withdrawals/expenses
                account_metrics[account_type] = {
                    'account_type': account_type,
                    'total_transactions': total_transactions,
                    'total_income': total_inflows,
                    'total_expenses': total_outflows,
                    'net_balance': net_change,
                    'interpretation': 'Positive balance = money available'
                }
        
        return account_metrics
    
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
