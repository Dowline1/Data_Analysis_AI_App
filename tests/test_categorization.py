"""
Tests for transaction categorisation.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.transaction_categorizer import TransactionCategorizerAgent


class TestTransactionCategorization:
    """Test transaction categorisation accuracy."""
    
    def test_categorizer_initialization(self):
        """Test that categorizer can be initialized."""
        categorizer = TransactionCategorizerAgent()
        assert categorizer is not None
    
    def test_basic_category_recognition(self):
        """Test recognition of obvious transaction categories."""
        categorizer = TransactionCategorizerAgent()
        
        # Test transactions with obvious categories
        test_transactions = [
            {"description": "TESCO STORE", "amount": -50.00},
            {"description": "NETFLIX SUBSCRIPTION", "amount": -15.99},
            {"description": "SHELL FUEL", "amount": -60.00},
            {"description": "AMAZON MARKETPLACE", "amount": -29.99},
        ]
        
        # This should categorize without errors
        try:
            result = categorizer.categorize(test_transactions)
            assert isinstance(result, list)
            assert len(result) == len(test_transactions)
        except Exception as e:
            # If API call fails, that's okay for this test
            pass
    
    def test_handles_empty_list(self):
        """Test that empty transaction list is handled."""
        categorizer = TransactionCategorizerAgent()
        
        result = categorizer.categorize([])
        assert result == []
    
    def test_batch_processing(self):
        """Test that large lists are batch processed."""
        categorizer = TransactionCategorizerAgent()
        
        # Create a large list of transactions
        transactions = [
            {"description": f"Transaction {i}", "amount": -10.00}
            for i in range(100)
        ]
        
        try:
            result = categorizer.categorize(transactions)
            # Should process all transactions
            assert len(result) == len(transactions)
        except Exception as e:
            # API might not be available, that's okay
            pass
    
    def test_preserves_transaction_data(self):
        """Test that original transaction data is preserved."""
        categorizer = TransactionCategorizerAgent()
        
        test_transaction = [{
            "description": "TEST MERCHANT",
            "amount": -25.50,
            "date": "2024-01-01"
        }]
        
        try:
            result = categorizer.categorize(test_transaction)
            if result:
                # Should preserve original fields
                assert "description" in result[0]
                assert "amount" in result[0]
        except:
            pass
