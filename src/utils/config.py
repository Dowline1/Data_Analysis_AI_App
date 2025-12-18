"""
Configuration management for the app.
Loads environment variables and provides app-wide settings.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class - basically just holds all our settings."""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-latest')
    GEMINI_TEMPERATURE = float(os.getenv('GEMINI_TEMPERATURE', '0.7'))
    
    # Paths
    DATA_DIR = Path('./data')
    OUTPUT_DIR = Path('./data/output')
    CACHE_DIR = Path('./data/cache')
    SAMPLE_STATEMENTS_DIR = DATA_DIR / 'sample_statements'
    
    # Application settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
    
    # Reflection settings
    MAX_REFLECTION_ITERATIONS = int(os.getenv('MAX_REFLECTION_ITERATIONS', '3'))
    
    # Transaction categorization keywords
    # Simple keyword matching for transaction categories
    CATEGORY_KEYWORDS = {
        'Groceries': ['tesco', 'supervalu', 'aldi', 'lidl', 'dunnes', 'spar', 'centra', 'grocery'],
        'Dining': ['restaurant', 'cafe', 'coffee', 'pizza', 'takeaway', 'uber eats', 'deliveroo'],
        'Transport': ['bus', 'luas', 'dart', 'taxi', 'uber', 'fuel', 'petrol', 'parking'],
        'Entertainment': ['cinema', 'spotify', 'netflix', 'amazon prime', 'disney', 'ticket'],
        'Bills': ['electric', 'gas', 'water', 'internet', 'phone', 'vodafone', 'three', 'eir'],
        'Shopping': ['amazon', 'ebay', 'asos', 'zara', 'penneys', 'h&m'],
        'Health': ['pharmacy', 'doctor', 'dentist', 'hospital', 'gym'],
        'Education': ['college', 'university', 'course', 'tuition'],
    }
    
    @classmethod
    def validate(cls):
        """Check that required settings are present."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        return True


# Validate config when imported
Config.validate()
