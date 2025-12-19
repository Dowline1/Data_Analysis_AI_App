"""
Guardrails for Data Ingestion
Implements security and safety checks:
1. File validation (size, type, malicious content)
2. PII masking (account numbers, sensitive data)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


class FileValidationGuardrail:
    """
    Guardrail #1: File validation and security checks.
    Prevents malicious files and enforces size/type limits.
    """
    
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    SUSPICIOUS_PATTERNS = [
        b'<script',  # JavaScript
        b'<?php',    # PHP code
        b'import os', # Python imports
        b'eval(',    # Code evaluation
    ]
    
    def __init__(self, max_size_mb: int = None):
        self.max_size_mb = max_size_mb or Config.MAX_FILE_SIZE_MB
        self.max_size_bytes = self.max_size_mb * 1024 * 1024
    
    def validate(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate file before processing.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            issues.append(f"File not found: {file_path}")
            return False, issues
        
        # Check file extension
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            issues.append(f"Invalid file type: {path.suffix}. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}")
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            issues.append(f"File too large: {size_mb:.2f}MB. Maximum: {self.max_size_mb}MB")
        
        if file_size == 0:
            issues.append("File is empty")
        
        # Check for suspicious content (basic security scan)
        try:
            with open(file_path, 'rb') as f:
                # Read first 10KB for security check
                sample = f.read(10240)
                
                for pattern in self.SUSPICIOUS_PATTERNS:
                    if pattern in sample.lower():
                        issues.append(f"Potentially malicious content detected: {pattern.decode('utf-8', errors='ignore')}")
        except Exception as e:
            issues.append(f"Unable to scan file content: {str(e)}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info(f"File validation passed: {file_path}")
        else:
            logger.warning(f"File validation failed: {', '.join(issues)}")
        
        return is_valid, issues


class PIIMaskingGuardrail:
    """
    Guardrail #2: PII (Personally Identifiable Information) masking.
    Protects sensitive data like account numbers, IBANs, etc.
    """
    
    # Regex patterns for common PII in bank statements
    PATTERNS = {
        'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b',  # IBAN format
        'account_number': r'\b\d{8,12}\b',  # 8-12 digit account numbers
        'card_last4': r'\*{4,}\d{4}\b',  # Card numbers like ****1234
        'sort_code': r'\b\d{2}-\d{2}-\d{2}\b',  # UK sort codes
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        'phone': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone numbers
    }
    
    # Masking strategies
    MASK_CHAR = '*'
    KEEP_LAST_N = 4  # Keep last 4 digits for reference
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.masked_count = {}
    
    def mask_transaction(self, transaction: Dict) -> Dict:
        """
        Mask PII in a single transaction.
        
        Args:
            transaction: Transaction dict with date, description, amount
            
        Returns:
            Transaction with PII masked
        """
        if not self.enabled:
            return transaction
        
        masked = transaction.copy()
        
        # Mask description field
        if 'description' in masked and masked['description']:
            masked['description'] = self._mask_text(masked['description'])
        
        return masked
    
    def mask_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """
        Mask PII in a list of transactions.
        
        Args:
            transactions: List of transaction dicts
            
        Returns:
            List of transactions with PII masked
        """
        if not self.enabled:
            return transactions
        
        self.masked_count = {}  # Reset count
        masked_transactions = [self.mask_transaction(t) for t in transactions]
        
        if any(self.masked_count.values()):
            logger.info(f"Masked PII: {self.masked_count}")
        
        return masked_transactions
    
    def _mask_text(self, text: str) -> str:
        """Apply PII masking to text."""
        masked_text = text
        
        # Apply each pattern
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, masked_text)
            
            for match in matches:
                original = match.group(0)
                masked = self._mask_value(original, pii_type)
                masked_text = masked_text.replace(original, masked)
                
                # Track what we masked
                self.masked_count[pii_type] = self.masked_count.get(pii_type, 0) + 1
        
        return masked_text
    
    def _mask_value(self, value: str, pii_type: str) -> str:
        """
        Mask a specific value based on its type.
        
        Keeps last 4 characters for reference, masks the rest.
        """
        if len(value) <= self.KEEP_LAST_N:
            # Too short to safely mask
            return self.MASK_CHAR * len(value)
        
        # Keep last N characters, mask the rest
        visible_part = value[-self.KEEP_LAST_N:]
        masked_part = self.MASK_CHAR * (len(value) - self.KEEP_LAST_N)
        
        return masked_part + visible_part
    
    def get_masking_summary(self) -> Dict[str, int]:
        """Get summary of what was masked."""
        return self.masked_count.copy()


class GuardrailManager:
    """
    Manages all guardrails for the application.
    Provides a single interface to apply all safety checks.
    """
    
    def __init__(self, enable_pii_masking: bool = True):
        self.file_validator = FileValidationGuardrail()
        self.pii_masker = PIIMaskingGuardrail(enabled=enable_pii_masking)
    
    def validate_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate file before processing (Guardrail #1).
        
        Returns:
            (is_valid, list_of_issues)
        """
        return self.file_validator.validate(file_path)
    
    def mask_pii(self, transactions: List[Dict]) -> List[Dict]:
        """
        Mask PII in transactions (Guardrail #2).
        
        Returns:
            Transactions with PII masked
        """
        return self.pii_masker.mask_transactions(transactions)
    
    def get_pii_masking_summary(self) -> Dict[str, int]:
        """Get summary of PII masking operations."""
        return self.pii_masker.get_masking_summary()
