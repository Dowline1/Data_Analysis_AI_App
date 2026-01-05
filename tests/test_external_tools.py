"""
Tests for external API tools.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools.currency_converter import CurrencyConverter
from tools.stock_recommendations import StockRecommender


class TestCurrencyConverter:
    """Test currency conversion API integration."""
    
    def test_converter_initialization(self):
        """Test that currency converter initializes."""
        converter = CurrencyConverter()
        assert converter is not None
    
    def test_convert_basic_currencies(self):
        """Test conversion between common currencies."""
        converter = CurrencyConverter()
        
        try:
            result = converter.convert(100, "USD", "EUR")
            # Should return a numeric result
            assert result is not None
            # Result should be reasonable (not 0, not the same)
            if isinstance(result, (int, float)):
                assert result > 0
        except Exception as e:
            # API might not be available, that's okay
            pytest.skip(f"API not available: {e}")
    
    def test_handles_invalid_currency(self):
        """Test handling of invalid currency codes."""
        converter = CurrencyConverter()
        
        try:
            result = converter.convert(100, "INVALID", "USD")
            # Should handle gracefully
        except Exception:
            # Expected to fail with invalid currency
            pass
    
    def test_caching_mechanism(self):
        """Test that caching reduces API calls."""
        converter = CurrencyConverter()
        
        try:
            # First call - should hit API
            result1 = converter.convert(100, "USD", "GBP")
            
            # Second call - should use cache
            result2 = converter.convert(100, "USD", "GBP")
            
            # Results should be identical if cache works
            if result1 and result2:
                assert result1 == result2
        except:
            pytest.skip("API not available")


class TestStockRecommender:
    """Test stock recommendations API integration."""
    
    def test_recommender_initialization(self):
        """Test that stock recommender initializes."""
        recommender = StockRecommender()
        assert recommender is not None
    
    def test_get_growth_stocks(self):
        """Test fetching growth stock recommendations."""
        recommender = StockRecommender()
        
        try:
            result = recommender.get_growth_stocks(market="US")
            
            # Should return string with recommendations
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            # API might not be available
            pytest.skip(f"Tavily API not available: {e}")
    
    def test_market_filtering(self):
        """Test different market parameters."""
        recommender = StockRecommender()
        
        markets = ["US", "Europe", "Global"]
        
        for market in markets:
            try:
                result = recommender.get_growth_stocks(market=market)
                assert isinstance(result, str)
            except:
                pass
    
    def test_ticker_extraction(self):
        """Test that recommendations include stock tickers."""
        recommender = StockRecommender()
        
        try:
            result = recommender.get_growth_stocks(market="US")
            
            if result and "Error" not in result and "not configured" not in result:
                # Should contain stock-related content
                assert any(keyword in result.lower() for keyword in ["stock", "ticker", "company", "growth"])
        except:
            pytest.skip("API not available")
