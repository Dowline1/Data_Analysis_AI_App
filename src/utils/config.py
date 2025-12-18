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
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
    
    # Paths
    DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
    OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './data/output'))
    CACHE_DIR = Path(os.getenv('CACHE_DIR', './data/cache'))
    SAMPLE_STATEMENTS_DIR = DATA_DIR / 'sample_statements'
    
    # Application settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    TIMEOUT_SECONDS = int(os.getenv('TIMEOUT_SECONDS', '30'))
    
    # Guardrails
    ENABLE_INPUT_VALIDATION = os.getenv('ENABLE_INPUT_VALIDATION', 'true').lower() == 'true'
    ENABLE_CODE_SAFETY = os.getenv('ENABLE_CODE_SAFETY', 'true').lower() == 'true'
    ENABLE_OUTPUT_VALIDATION = os.getenv('ENABLE_OUTPUT_VALIDATION', 'true').lower() == 'true'
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
    
    # HITL settings
    HITL_ENABLED = os.getenv('HITL_ENABLED', 'true').lower() == 'true'
    HITL_TIMEOUT_SECONDS = int(os.getenv('HITL_TIMEOUT_SECONDS', '300'))
    
    # Reflection settings
    MAX_REFLECTION_ITERATIONS = int(os.getenv('MAX_REFLECTION_ITERATIONS', '3'))
    REFLECTION_ENABLED = os.getenv('REFLECTION_ENABLED', 'true').lower() == 'true'
    
    # Development
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    SAVE_INTERMEDIATE_STATES = os.getenv('SAVE_INTERMEDIATE_STATES', 'true').lower() == 'true'
    
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
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        return True


# Validate config when imported
Config.validate()
