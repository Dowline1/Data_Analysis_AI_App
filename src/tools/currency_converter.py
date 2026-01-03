"""
Currency Conversion Tool - External API Integration

Uses ExchangeRate-API to fetch real-time currency exchange rates.
This is an external tool that demonstrates API integration for the assignment.
"""

import os
import requests
from typing import Dict, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CurrencyConverter:
    """
    External API tool for currency conversion using real-time exchange rates.
    
    Uses the free tier of ExchangeRate-API (https://www.exchangerate-api.com/)
    which provides 1,500 requests/month without requiring an API key.
    """
    
    BASE_URL = "https://api.exchangerate-api.com/v4/latest"
    
    # Common currencies for the dropdown
    SUPPORTED_CURRENCIES = [
        ("EUR", "Euro (€)"),
        ("USD", "US Dollar ($)"),
        ("GBP", "British Pound (£)"),
        ("JPY", "Japanese Yen (¥)"),
        ("CHF", "Swiss Franc (Fr)"),
        ("CAD", "Canadian Dollar ($)"),
        ("AUD", "Australian Dollar ($)"),
        ("CNY", "Chinese Yuan (¥)"),
        ("INR", "Indian Rupee (₹)"),
        ("SEK", "Swedish Krona (kr)"),
        ("NOK", "Norwegian Krone (kr)"),
        ("DKK", "Danish Krone (kr)"),
    ]
    
    def __init__(self):
        """Initialize the currency converter."""
        self.cache = {}  # Cache rates to minimize API calls
        logger.info("Currency converter initialized with external API")
    
    def get_exchange_rate(self, base_currency: str, target_currency: str) -> Optional[float]:
        """
        Get the exchange rate from base currency to target currency.
        
        Args:
            base_currency: Source currency code (e.g., 'EUR')
            target_currency: Target currency code (e.g., 'USD')
            
        Returns:
            Exchange rate as float, or None if request fails
        """
        # Same currency, no conversion needed
        if base_currency == target_currency:
            return 1.0
        
        # Check cache first
        cache_key = f"{base_currency}_{target_currency}"
        if cache_key in self.cache:
            logger.info(f"Using cached rate for {cache_key}")
            return self.cache[cache_key]
        
        try:
            logger.info(f"Fetching exchange rate: {base_currency} to {target_currency}")
            
            # Call external API
            response = requests.get(
                f"{self.BASE_URL}/{base_currency}",
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            rates = data.get("rates", {})
            
            if target_currency in rates:
                rate = rates[target_currency]
                self.cache[cache_key] = rate
                logger.info(f"Exchange rate: 1 {base_currency} = {rate} {target_currency}")
                return rate
            else:
                logger.error(f"Target currency {target_currency} not found in response")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch exchange rate: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in currency conversion: {e}")
            return None
    
    def convert_amount(
        self, 
        amount: float, 
        from_currency: str, 
        to_currency: str
    ) -> Optional[float]:
        """
        Convert an amount from one currency to another.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Converted amount, or None if conversion fails
        """
        rate = self.get_exchange_rate(from_currency, to_currency)
        if rate is not None:
            converted = amount * rate
            logger.info(f"Converted {amount} {from_currency} to {converted:.2f} {to_currency}")
            return converted
        return None
    
    def get_all_rates(self, base_currency: str) -> Optional[Dict[str, float]]:
        """
        Get all exchange rates for a base currency.
        
        Args:
            base_currency: Base currency code (e.g., 'EUR')
            
        Returns:
            Dictionary of currency codes to exchange rates
        """
        try:
            logger.info(f"Fetching all rates for base currency: {base_currency}")
            
            response = requests.get(
                f"{self.BASE_URL}/{base_currency}",
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            rates = data.get("rates", {})
            
            logger.info(f"Fetched {len(rates)} exchange rates")
            return rates
            
        except Exception as e:
            logger.error(f"Failed to fetch exchange rates: {e}")
            return None


def convert_currency_tool(amount: float, from_currency: str, to_currency: str) -> str:
    """
    LangChain tool function for currency conversion.
    
    This is the tool function that will be exposed to the ReAct agent.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., 'EUR')
        to_currency: Target currency code (e.g., 'USD')
        
    Returns:
        Formatted string with conversion result
    """
    converter = CurrencyConverter()
    result = converter.convert_amount(amount, from_currency, to_currency)
    
    if result is not None:
        return f"{amount:.2f} {from_currency} = {result:.2f} {to_currency}"
    else:
        return f"Failed to convert {from_currency} to {to_currency}. Please check currency codes."
