"""
Transaction Categorizer Agent - Hybrid Pattern
Uses keyword matching first, then LLM for dynamic categorization.
"""

from typing import List, Dict
import pandas as pd
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.state.app_state import Transaction

logger = setup_logger(__name__)


class CategoryResult(BaseModel):
    """Schema for LLM categorization response."""
    category: str = Field(description="The category name (e.g., Groceries, Dining, Transport, etc.)")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class TransactionCategorizerAgent:
    """
    Hybrid agent that uses keyword matching first, then LLM for dynamic categorization.
    Works with any transaction description format.
    """
    
    def __init__(self):
        self.category_keywords = Config.CATEGORY_KEYWORDS
        self.unknown_category = "Other"
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.3
        )
        self.llm_cache = {}  # Cache LLM categorizations
        
        # Standard categories for LLM to choose from
        self.standard_categories = [
            'Groceries', 'Dining', 'Transport', 'Entertainment', 'Bills',
            'Shopping', 'Health', 'Education', 'Income', 'Transfers',
            'Subscriptions', 'Insurance', 'Housing', 'Savings', 'Other'
        ]
    
    def categorize_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        """
        Categorize all transactions using hybrid approach:
        1. Try keyword matching first (fast)
        2. Use LLM for unmatched transactions (accurate)
        
        Args:
            transactions: List of transaction dicts
            
        Returns:
            List of transactions with 'category' field populated
        """
        logger.info(f"Categorizing {len(transactions)} transactions using hybrid approach")
        
        categorized = []
        category_counts = {}
        keyword_matched = 0
        llm_categorized = 0
        
        # Collect transactions that need LLM categorization
        llm_batch = []
        
        for transaction in transactions:
            # Try keyword matching first
            category = self._keyword_match(transaction)
            
            if category != self.unknown_category:
                # Keyword match successful
                transaction['category'] = category
                categorized.append(transaction)
                keyword_matched += 1
            else:
                # Need LLM categorization
                llm_batch.append(transaction)
            
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Batch process LLM categorizations if needed
        if llm_batch:
            logger.info(f"Using LLM to categorize {len(llm_batch)} unmatched transactions")
            for transaction in llm_batch:
                category = self._llm_categorize(transaction)
                transaction['category'] = category
                categorized.append(transaction)
                category_counts[category] = category_counts.get(category, 0) + 1
                llm_categorized += 1
        
        logger.info(f"Categorization complete: {keyword_matched} keyword-matched, {llm_categorized} LLM-categorized")
        logger.info(f"Category distribution: {category_counts}")
        
        return categorized
    
    def _keyword_match(self, transaction: Transaction) -> str:
        """
        Fast keyword matching against predefined categories.
        
        Args:
            transaction: Single transaction dict
            
        Returns:
            Category name or "Other" if no match
        """
        description = transaction.get('description', '').lower()
        amount = transaction.get('amount', 0)
        
        # Check each category's keywords
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in description:
                    logger.debug(f"Keyword match: '{description[:50]}' -> '{category}'")
                    return category
        
        # Special logic: Check transaction type by amount
        if amount > 0:
            if 'transfer from' in description or 'top-up' in description or 'deposit' in description:
                return "Income"
        else:
            if 'transfer to' in description:
                return "Transfers"
        
        return self.unknown_category
    
    def _llm_categorize(self, transaction: Transaction) -> str:
        """
        Use LLM to intelligently categorize transaction based on description.
        Works dynamically with any transaction description.
        
        Args:
            transaction: Single transaction dict
            
        Returns:
            Category name
        """
        description = transaction.get('description', '')
        amount = transaction.get('amount', 0)
        
        # Check cache first
        cache_key = description.lower()
        if cache_key in self.llm_cache:
            logger.debug(f"Using cached category for: '{description[:50]}'")
            return self.llm_cache[cache_key]
        
        # Prepare prompt for LLM
        parser = JsonOutputParser(pydantic_object=CategoryResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial transaction categorizer. Analyze the transaction description and categorize it.

Available categories:
- Groceries: supermarkets, food shopping
- Dining: restaurants, cafes, food delivery
- Transport: public transport, taxis, fuel, parking
- Entertainment: streaming services, movies, games, hobbies
- Bills: utilities, phone, internet
- Shopping: retail, online shopping, clothing
- Health: pharmacy, gym, medical
- Education: courses, books, tuition
- Income: salary, payments received
- Transfers: money transfers between accounts
- Subscriptions: recurring monthly/annual services
- Insurance: health, car, home insurance
- Housing: rent, mortgage, maintenance
- Savings: savings deposits, investments
- Other: anything that doesn't fit above

Consider the transaction amount ({amount}) to help identify if it's income (positive) or expense (negative).

{format_instructions}"""),
            ("user", "Categorize this transaction:\nDescription: {description}\nAmount: {amount}")
        ])
        
        try:
            chain = prompt | self.llm | parser
            result = chain.invoke({
                'description': description,
                'amount': amount,
                'format_instructions': parser.get_format_instructions()
            })
            
            category = result.get('category', self.unknown_category)
            
            # Validate category is in our standard list
            if category not in self.standard_categories:
                # Find closest match or use Other
                category_lower = category.lower()
                for std_cat in self.standard_categories:
                    if std_cat.lower() in category_lower or category_lower in std_cat.lower():
                        category = std_cat
                        break
                else:
                    category = self.unknown_category
            
            # Cache the result
            self.llm_cache[cache_key] = category
            
            logger.debug(f"LLM categorized '{description[:50]}' as '{category}'")
            return category
            
        except Exception as e:
            logger.error(f"LLM categorization failed for '{description}': {e}")
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
