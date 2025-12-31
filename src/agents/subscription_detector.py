"""
Subscription Detector Agent - CodeAct Pattern
Executes Python code to detect recurring transactions (subscriptions).
"""

from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.graph.state import Transaction

logger = setup_logger(__name__)


class SubscriptionValidation(BaseModel):
    """Schema for LLM subscription validation."""
    is_subscription: bool = Field(description="True if this is a real subscription/recurring service")
    reason: str = Field(description="Brief explanation of why this is or isn't a subscription")


class SubscriptionDetectorAgent:
    """
    CodeAct agent that executes Python code to detect recurring transactions.
    Uses pattern matching and temporal analysis to identify subscriptions.
    """
    
    def __init__(self, min_occurrences: int = 3, max_day_variance: int = 3, use_llm_validation: bool = True):
        """
        Initialize detector.
        
        Args:
            min_occurrences: Minimum times a transaction must repeat to be considered recurring (default 3)
            max_day_variance: Maximum variance in days between occurrences (default 3)
            use_llm_validation: Use LLM to validate if patterns are real subscriptions (default True)
        """
        self.min_occurrences = min_occurrences
        self.max_day_variance = max_day_variance
        self.use_llm_validation = use_llm_validation

        # Categories to exclude from subscription detection unless keyword matches
        self.excluded_categories = [
            'Groceries', 'Dining', 'Restaurants', 'Supermarket', 'Fuel', 'ATM', 'Transfer',
            'Hardware', 'Home Improvement', 'Travel', 'Hotel', 'Airline', 'Medical', 'Pharmacy',
            'Education', 'Tuition', 'Gift', 'Charity', 'Tax', 'Government', 'Insurance', 'Loan',
            'Mortgage', 'Rent', 'Credit Card Payment', 'Bank Fee', 'Cash', 'Withdrawal', 'Deposit',
            'Refund', 'Salary', 'Income', 'Bonus', 'Investment', 'Stock', 'Bond', 'Dividend',
            'Utilities', 'Electricity', 'Gas', 'Water', 'Internet', 'Mobile', 'Phone', 'Cable',
            'Entertainment', 'Sports', 'Leisure', 'Miscellaneous', 'Other'
        ]
        
        # Hard exclusions - never treat these as subscriptions regardless of patterns
        self.hard_exclusions = [
            'mortgage', 'rent', 'loan', 'credit card', 'bank fee', 'atm', 'withdrawal', 'deposit',
            'transfer', 'tax', 'insurance', 'salary', 'payroll', 'refund', 'cashback',
            'tuition', 'school fee', 'utility', 'electricity', 'gas bill', 'water bill'
        ]

        # Keywords that indicate a subscription service
        self.subscription_keywords = [
            'subscription', 'recurring', 'monthly', 'annual', 'service', 'membership', 'plan',
            'netflix', 'spotify', 'apple', 'amazon', 'prime', 'google', 'microsoft', 'adobe',
            'cloud', 'tv', 'music', 'video', 'gym', 'fitness', 'magazine', 'newspaper', 'news',
            'insurance', 'storage', 'dropbox', 'office', '365', 'zoom', 'discord', 'slack', 'hulu',
            'disney', 'paramount', 'crunchyroll', 'audible', 'kindle', 'playstation', 'xbox', 'nintendo',
            'canva', 'patreon', 'substack', 'linkedin', 'coursera', 'udemy', 'masterclass', 'calm', 'headspace'
        ]
        
        # Initialize LLM for intelligent validation
        if self.use_llm_validation:
            self.llm = ChatGoogleGenerativeAI(
                model=Config.GEMINI_MODEL,
                google_api_key=Config.GOOGLE_API_KEY,
                temperature=0.3
            )
            logger.info("Subscription detector using LLM validation for intelligent filtering")
        else:
            self.llm = None
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _llm_confirms_subscription(
        self,
        description: str,
        category: str,
        occurrences: int,
        amount_mean: float,
        amount_std: float,
        dates: List[str],
        intervals: List[int],
        reason: str,
        has_keyword: bool
    ) -> bool:
        """Use LLM to validate borderline cases with strong subscription signals."""
        if not (self.use_llm_validation and self.llm and has_keyword):
            return False

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a financial assistant that identifies recurring subscriptions. "
                "ONLY respond with valid JSON matching this schema: {{\"is_subscription\": bool, \"reason\": str}}."
            ),
            (
                "human",
                "Description: {description}\n"
                "Category: {category}\n"
                "Occurrences: {occurrences}\n"
                "Amount mean: {amount_mean}\n"
                "Amount std: {amount_std}\n"
                "Dates: {dates}\n"
                "Intervals (days): {intervals}\n"
                "Reason for review: {reason}\n"
                "Is this a true subscription/recurring service?"
            )
        ])

        parser = JsonOutputParser(pydantic_object=SubscriptionValidation)
        chain = prompt | self.llm | parser

        payload = {
            'description': description,
            'category': category or 'Unknown',
            'occurrences': occurrences,
            'amount_mean': f"{amount_mean:.2f}",
            'amount_std': f"{amount_std:.2f}",
            'dates': dates,
            'intervals': intervals if intervals else ['N/A'],
            'reason': reason
        }

        try:
            validation = chain.invoke(payload)
        except Exception as exc:
            logger.warning(f"LLM validation failed for '{description}': {exc}")
            return False

        if isinstance(validation, dict):
            validation = SubscriptionValidation(**validation)

        if validation.is_subscription:
            logger.debug(f"LLM validated '{description}' as subscription: {validation.reason}")
            return True

        logger.debug(f"LLM rejected '{description}' as subscription: {validation.reason}")
        return False
    
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
        # Group by description (don't round amount - subscriptions should be exact)
        grouped = expenses.groupby('description')
        
        for description, group in grouped:
            occurrences = len(group)
            if occurrences < self.min_occurrences:
                continue

            category = group['category'].iloc[0] if 'category' in group.columns else 'Other'
            desc_lower = description.lower()
            has_subscription_keyword = any(keyword in desc_lower for keyword in self.subscription_keywords)

            # Filter 1: Hard exclusions - never treat these as subscriptions
            if any(exclusion in desc_lower for exclusion in self.hard_exclusions):
                logger.debug(f"Skipping '{description}' - matches hard exclusion list")
                continue

            # Filter 2: Skip categories typically not subscriptions (unless keyword match)
            if category in self.excluded_categories and not has_subscription_keyword:
                continue

            # Filter 3: Skip restaurants/dining (even if they recur)
            restaurant_keywords = ['restaurant', 'tavern', 'grill', 'cafe', 'coffee',
                                   'pizza', 'burger', 'bbq', 'diner', 'bistro', 'bar']
            if any(keyword in desc_lower for keyword in restaurant_keywords):
                continue

            amounts = group['amount'].values
            amount_std = amounts.std()
            amount_mean = amounts.mean()
            validated_by_llm = False

            dates = sorted(group['date'].tolist())
            str_dates = [str(d.date()) for d in dates]
            if len(str_dates) < 2:
                continue

            intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            if not intervals:
                continue

            variance_ratio = abs(amount_std) / (abs(amount_mean) + 1e-9)
            if variance_ratio > 0.30:
                reason = f"Amount variance ratio {variance_ratio:.2f} exceeds threshold"
                if self._llm_confirms_subscription(
                    description,
                    category,
                    occurrences,
                    amount_mean,
                    amount_std,
                    str_dates,
                    intervals,
                    reason,
                    has_subscription_keyword
                ):
                    validated_by_llm = True
                else:
                    continue

            avg_interval = sum(intervals) / len(intervals)
            max_variance = max(abs(interval - avg_interval) for interval in intervals)

            if max_variance > self.max_day_variance:
                reason = (
                    f"Intervals vary by up to {max_variance:.1f} days (avg {avg_interval:.1f} days), "
                    f"exceeding allowed {self.max_day_variance} days"
                )
                if self._llm_confirms_subscription(
                    description,
                    category,
                    occurrences,
                    amount_mean,
                    amount_std,
                    str_dates,
                    intervals,
                    reason,
                    has_subscription_keyword
                ):
                    validated_by_llm = True
                else:
                    continue

            frequency = self._determine_frequency(avg_interval)
            amount = amount_mean
            monthly_cost = self._calculate_monthly_cost(abs(amount), avg_interval)

            subscription = {
                'description': description,
                'amount': float(amount),
                'frequency': frequency,
                'interval_days': round(avg_interval, 1),
                'occurrences': occurrences,
                'first_seen': str(dates[0].date()),
                'last_seen': str(dates[-1].date()),
                'estimated_monthly_cost': round(monthly_cost, 2),
                'category': category,
                'is_regular': max_variance <= 2,
                'amount_variance': round(float(amount_std), 2),
                'validated_by_llm': validated_by_llm
            }

            subscriptions.append(subscription)
            logger.debug(f"Detected subscription: {description} - {frequency} (\u20ac{monthly_cost:.2f}/month)")
        
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
