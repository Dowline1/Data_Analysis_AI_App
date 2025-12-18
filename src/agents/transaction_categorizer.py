"""
Transaction Categorizer Agent - CodeAct Pattern
Executes Python code to categorize transactions using keyword matching.
"""

from typing import List, Dict
import pandas as pd
from datetime import datetime

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.state.app_state import Transaction

logger = setup_logger(__name__)


class TransactionCategorizerAgent:
    """
    CodeAct agent that executes Python code to categorize transactions.
    Uses keyword matching against predefined categories.
    """
    
    def __init__(self):
        self.category_keywords = Config.CATEGORY_KEYWORDS
        self.unknown_category = "Other"
    
    def categorize_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        """
        Categorize all transactions using Python code execution (CodeAct pattern).
        
        Args:
            transactions: List of transaction dicts
            
        Returns:
            List of transactions with 'category' field populated
        """
        logger.info(f"Categorizing {len(transactions)} transactions using Python keyword matching")
        
        categorized = []
        category_counts = {}
        
        for transaction in transactions:
            # Execute Python code to find matching category
            category = self._execute_categorization(transaction)
            
            # Update transaction with category
            transaction['category'] = category
            categorized.append(transaction)
            
            # Track category distribution
            category_counts[category] = category_counts.get(category, 0) + 1
        
        logger.info(f"Categorization complete: {category_counts}")
        
        return categorized
    
    def _execute_categorization(self, transaction: Transaction) -> str:
        """
        Execute Python code to determine transaction category.
        
        This is the CodeAct pattern - executing actual Python logic
        to process the data.
        
        Args:
            transaction: Single transaction dict
            
        Returns:
            Category name
        """
        description = transaction.get('description', '').lower()
        amount = transaction.get('amount', 0)
        
        # Execute Python logic: keyword matching
        # Priority-based matching: more specific categories first
        
        # Check each category's keywords
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in description:
                    logger.debug(f"Matched '{description[:50]}' to '{category}' via keyword '{keyword}'")
                    return category
        
        # Special logic: Check transaction type by amount
        if amount > 0:
            # Positive amount without matching keywords -> Income
            if 'transfer from' in description or 'top-up' in description or 'deposit' in description:
                return "Income"
        else:
            # Negative amount without matching keywords -> Expense
            if 'transfer to' in description:
                return "Transfers"
        
        # No match found
        logger.debug(f"No category match for: '{description[:50]}'")
        return self.unknown_category
    
    def get_category_summary(self, transactions: List[Transaction]) -> Dict:
        """
        Execute Python code to generate category summary statistics.
        
        CodeAct pattern: Using pandas for efficient computation.
        
        Args:
            transactions: List of categorized transactions
            
        Returns:
            Dict with category statistics
        """
        logger.info("Executing Python code to calculate category statistics")
        
        # Convert to DataFrame for efficient computation (CodeAct)
        df = pd.DataFrame(transactions)
        
        # Execute pandas operations
        category_summary = df.groupby('category').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        # Convert to dict format
        summary = {}
        for category in category_summary.index:
            summary[category] = {
                'total': float(category_summary.loc[category, ('amount', 'sum')]),
                'count': int(category_summary.loc[category, ('amount', 'count')]),
                'average': float(category_summary.loc[category, ('amount', 'mean')])
            }
        
        logger.info(f"Category summary calculated for {len(summary)} categories")
        
        return summary
    
    def get_spending_by_category(self, transactions: List[Transaction]) -> Dict[str, float]:
        """
        Execute Python code to get total spending by category.
        
        Args:
            transactions: List of categorized transactions
            
        Returns:
            Dict mapping category to total spending (negative amounts only)
        """
        # CodeAct: Execute Python logic
        spending = {}
        
        for transaction in transactions:
            category = transaction.get('category', self.unknown_category)
            amount = transaction.get('amount', 0)
            
            # Only count expenses (negative amounts)
            if amount < 0:
                spending[category] = spending.get(category, 0) + abs(amount)
        
        # Sort by spending amount
        spending = dict(sorted(spending.items(), key=lambda x: x[1], reverse=True))
        
        return spending
