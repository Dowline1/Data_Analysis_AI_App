"""
Subscription Detector Agent - CodeAct Pattern
Executes Python code to detect recurring transactions (subscriptions).
"""

from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from src.utils.logger import setup_logger
from src.state.app_state import Transaction

logger = setup_logger(__name__)


class SubscriptionDetectorAgent:
    """
    CodeAct agent that executes Python code to detect recurring transactions.
    Uses pattern matching and temporal analysis to identify subscriptions.
    """
    
    def __init__(self, min_occurrences: int = 2, max_day_variance: int = 5):
        """
        Initialize detector.
        
        Args:
            min_occurrences: Minimum times a transaction must repeat to be considered recurring
            max_day_variance: Maximum variance in days between occurrences
        """
        self.min_occurrences = min_occurrences
        self.max_day_variance = max_day_variance
    
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
    
    def detect_subscriptions(self, transactions: List[Transaction]) -> Dict:
        """
        Execute Python code to detect recurring transactions.
        
        CodeAct Pattern: Executes temporal pattern analysis using pandas.
        
        Args:
            transactions: List of categorized transactions
            
        Returns:
            Dict with detected subscriptions and analysis
        """
        logger.info(f"Detecting subscriptions in {len(transactions)} transactions")
        
        if not transactions:
            return {'subscriptions': [], 'total_subscription_cost': 0.0, 'count': 0}
        
        # Convert to DataFrame (CodeAct)
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        
        # Execute subscription detection algorithm
        subscriptions = self._execute_detection_algorithm(df)
        
        # Calculate total monthly cost
        total_cost = sum(sub['estimated_monthly_cost'] for sub in subscriptions)
        
        result = {
            'subscriptions': subscriptions,
            'total_subscription_cost': round(total_cost, 2),
            'count': len(subscriptions)
        }
        
        # Convert numpy types to native Python types for serialization
        result = self._convert_numpy_types(result)
        
        logger.info(f"Detected {len(subscriptions)} subscriptions, total monthly cost: €{total_cost:.2f}")
        
        return result
    
    def _execute_detection_algorithm(self, df: pd.DataFrame) -> List[Dict]:
        """
        Execute Python code for subscription detection algorithm.
        
        Algorithm:
        1. Group similar transactions (same description and similar amount)
        2. Check if they occur at regular intervals
        3. Calculate frequency and cost
        
        Args:
            df: DataFrame of transactions
            
        Returns:
            List of detected subscription dicts
        """
        subscriptions = []
        
        # Filter to expenses only
        expenses = df[df['amount'] < 0].copy()
        
        if len(expenses) == 0:
            return subscriptions
        
        # Execute grouping logic
        # Group by description and round amount to handle small variations
        expenses['amount_rounded'] = expenses['amount'].round(0)
        
        # Group similar transactions
        grouped = expenses.groupby(['description', 'amount_rounded'])
        
        for (description, amount), group in grouped:
            if len(group) < self.min_occurrences:
                continue
            
            # Execute temporal analysis
            dates = sorted(group['date'].tolist())
            
            # Calculate intervals between occurrences
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            
            if not intervals:
                continue
            
            # Check if intervals are roughly consistent (recurring pattern)
            avg_interval = sum(intervals) / len(intervals)
            max_variance = max(abs(interval - avg_interval) for interval in intervals)
            
            if max_variance <= self.max_day_variance:
                # This is a recurring transaction!
                
                # Determine frequency
                frequency = self._determine_frequency(avg_interval)
                
                # Calculate monthly cost estimate
                monthly_cost = self._calculate_monthly_cost(abs(amount), avg_interval)
                
                # Get category
                category = group['category'].iloc[0] if 'category' in group.columns else 'Other'
                
                subscription = {
                    'description': description,
                    'amount': float(amount),
                    'frequency': frequency,
                    'interval_days': round(avg_interval, 1),
                    'occurrences': len(group),
                    'first_seen': str(dates[0].date()),
                    'last_seen': str(dates[-1].date()),
                    'estimated_monthly_cost': round(monthly_cost, 2),
                    'category': category,
                    'is_regular': max_variance <= 2  # Very regular if variance < 2 days
                }
                
                subscriptions.append(subscription)
                
                logger.debug(f"Detected subscription: {description} - {frequency} (€{monthly_cost:.2f}/month)")
        
        # Sort by monthly cost
        subscriptions.sort(key=lambda x: x['estimated_monthly_cost'], reverse=True)
        
        return subscriptions
    
    def _determine_frequency(self, avg_interval: float) -> str:
        """
        Execute Python logic to determine frequency label.
        
        Args:
            avg_interval: Average days between occurrences
            
        Returns:
            Frequency string
        """
        if avg_interval <= 1:
            return "Daily"
        elif avg_interval <= 7:
            return "Weekly"
        elif avg_interval <= 10:
            return "Weekly (approx)"
        elif avg_interval <= 16:
            return "Bi-weekly"
        elif avg_interval <= 31:
            return "Monthly"
        elif avg_interval <= 93:
            return "Quarterly"
        elif avg_interval <= 186:
            return "Semi-annual"
        else:
            return "Annual"
    
    def _calculate_monthly_cost(self, amount: float, interval_days: float) -> float:
        """
        Execute Python calculation for estimated monthly cost.
        
        Args:
            amount: Transaction amount (absolute)
            interval_days: Average days between occurrences
            
        Returns:
            Estimated monthly cost
        """
        if interval_days == 0:
            return 0.0
        
        # Calculate how many times per month
        occurrences_per_month = 30 / interval_days
        
        return amount * occurrences_per_month
    
    def get_subscription_insights(self, subscriptions: List[Dict]) -> Dict:
        """
        Execute Python code to generate insights about subscriptions.
        
        Args:
            subscriptions: List of detected subscriptions
            
        Returns:
            Dict with insights
        """
        if not subscriptions:
            return {
                'total_monthly': 0.0,
                'total_annual': 0.0,
                'most_expensive': None,
                'by_frequency': {},
                'by_category': {}
            }
        
        # Execute pandas aggregation
        df = pd.DataFrame(subscriptions)
        
        total_monthly = df['estimated_monthly_cost'].sum()
        
        # Group by frequency
        by_frequency = df.groupby('frequency')['estimated_monthly_cost'].sum().to_dict()
        
        # Group by category
        by_category = df.groupby('category')['estimated_monthly_cost'].sum().to_dict()
        
        # Most expensive
        most_expensive = df.loc[df['estimated_monthly_cost'].idxmax()].to_dict()
        
        return {
            'total_monthly': round(total_monthly, 2),
            'total_annual': round(total_monthly * 12, 2),
            'most_expensive': {
                'description': most_expensive['description'],
                'monthly_cost': most_expensive['estimated_monthly_cost'],
                'frequency': most_expensive['frequency']
            },
            'by_frequency': {k: round(v, 2) for k, v in by_frequency.items()},
            'by_category': {k: round(v, 2) for k, v in by_category.items()}
        }
