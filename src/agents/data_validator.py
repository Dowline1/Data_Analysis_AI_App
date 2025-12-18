"""
Data Validation Agent - ReAct Pattern
Validates extracted transaction data quality and provides feedback for improvement.
"""

from typing import List, Dict, Any
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.state.app_state import Transaction

logger = setup_logger(__name__)


class ValidationResult(BaseModel):
    """Structured validation result from LLM."""
    quality_score: float = Field(description="Overall quality score 0-100")
    is_valid: bool = Field(description="Whether data meets minimum quality threshold")
    issues_found: List[str] = Field(description="List of specific issues identified")
    suggestions: List[str] = Field(description="Suggestions for improvement")
    reflection: str = Field(description="LLM's reasoning about the data quality")


class DataValidatorAgent:
    """
    ReAct agent that validates transaction data quality.
    Uses LLM to reason about data issues and provide actionable feedback.
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=0.3,  # Lower temperature for consistent validation
            google_api_key=Config.GOOGLE_API_KEY
        )
        
        self.validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data quality expert validating financial transaction data.

Your task is to analyze the provided transactions and assess data quality.

VALIDATION CRITERIA:
1. Completeness: Are all required fields present (date, description, amount)?
2. Date validity: Are dates in proper format and reasonable?
3. Amount validity: Are amounts numeric and non-zero?
4. Duplicates: Are there suspicious duplicate transactions?
5. Consistency: Is the data internally consistent?
6. Reasonableness: Do transactions make logical sense?

QUALITY SCORING:
- 90-100: Excellent quality, ready for analysis
- 70-89: Good quality, minor issues
- 50-69: Moderate issues, needs attention
- Below 50: Poor quality, requires re-extraction

You MUST respond with ONLY valid JSON matching this structure:
{{
    "quality_score": <number 0-100>,
    "is_valid": <boolean>,
    "issues_found": [<list of strings>],
    "suggestions": [<list of strings>],
    "reflection": "<your reasoning as a single string>"
}}

Do NOT use markdown formatting, bullet points, or extra text. ONLY JSON."""),
            ("human", """Analyze these {transaction_count} transactions:

{transactions_summary}

Provide validation results as JSON only.""")
        ])
        
        self.parser = JsonOutputParser(pydantic_object=ValidationResult)
        self.chain = self.validation_prompt | self.llm | self.parser
        
        # Minimum quality threshold
        self.min_quality_score = 70
    
    def validate(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """
        Validate transaction data quality.
        
        Args:
            transactions: List of extracted transactions
            
        Returns:
            Dict with validation results, stats, and reflection
        """
        logger.info(f"Validating {len(transactions)} transactions")
        
        # Calculate basic statistics
        stats = self._calculate_stats(transactions)
        
        # Prepare transactions summary for LLM
        transactions_summary = self._prepare_summary(transactions)
        
        # Get LLM validation assessment
        try:
            logger.info("Calling LLM for quality assessment...")
            llm_result = self.chain.invoke({
                "transaction_count": len(transactions),
                "transactions_summary": transactions_summary
            })
            
            logger.info(f"LLM Quality Score: {llm_result['quality_score']}/100")
            logger.info(f"Data Valid: {llm_result['is_valid']}")
            
            # Combine stats and LLM results
            validation_result = {
                "is_valid": llm_result['is_valid'],
                "quality_score": llm_result['quality_score'],
                "statistics": stats,
                "issues": llm_result['issues_found'],
                "suggestions": llm_result['suggestions'],
                "reflection": llm_result['reflection'],
                "needs_reextraction": llm_result['quality_score'] < self.min_quality_score
            }
            
            if validation_result['needs_reextraction']:
                logger.warning(f"Quality score {llm_result['quality_score']} below threshold {self.min_quality_score}")
            else:
                logger.info("Data validation passed")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            
            # Fallback to rule-based validation
            logger.info("Falling back to rule-based validation")
            return self._fallback_validation(transactions, stats)
    
    def _calculate_stats(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Calculate statistical metrics about the transactions."""
        if not transactions:
            return {
                "total_count": 0,
                "complete_count": 0,
                "missing_dates": 0,
                "missing_descriptions": 0,
                "missing_amounts": 0,
                "zero_amounts": 0,
                "completeness_rate": 0.0
            }
        
        total = len(transactions)
        missing_dates = sum(1 for t in transactions if not t.get('date'))
        missing_descriptions = sum(1 for t in transactions if not t.get('description'))
        missing_amounts = sum(1 for t in transactions if t.get('amount') is None)
        zero_amounts = sum(1 for t in transactions if t.get('amount') == 0)
        
        complete = total - max(missing_dates, missing_descriptions, missing_amounts)
        
        return {
            "total_count": total,
            "complete_count": complete,
            "missing_dates": missing_dates,
            "missing_descriptions": missing_descriptions,
            "missing_amounts": missing_amounts,
            "zero_amounts": zero_amounts,
            "completeness_rate": round((complete / total) * 100, 2) if total > 0 else 0.0
        }
    
    def _prepare_summary(self, transactions: List[Transaction]) -> str:
        """Prepare a summary of transactions for LLM analysis."""
        # Show first 10 and last 5 transactions
        sample_size = min(15, len(transactions))
        
        if len(transactions) <= 15:
            sample = transactions
        else:
            sample = transactions[:10] + transactions[-5:]
        
        summary_lines = []
        for i, t in enumerate(sample, 1):
            date = t.get('date', 'MISSING')
            desc = t.get('description', 'MISSING')[:50]
            amount = t.get('amount', 'MISSING')
            
            summary_lines.append(f"{i}. {date} | €{amount} | {desc}")
        
        if len(transactions) > 15:
            summary_lines.insert(10, f"... ({len(transactions) - 15} more transactions) ...")
        
        return "\n".join(summary_lines)
    
    def _fallback_validation(self, transactions: List[Transaction], stats: Dict) -> Dict[str, Any]:
        """Rule-based validation fallback if LLM fails."""
        completeness = stats['completeness_rate']
        
        issues = []
        suggestions = []
        
        if stats['missing_dates'] > 0:
            issues.append(f"{stats['missing_dates']} transactions missing dates")
            suggestions.append("Re-extract with focus on date column identification")
        
        if stats['missing_descriptions'] > 0:
            issues.append(f"{stats['missing_descriptions']} transactions missing descriptions")
            suggestions.append("Check if description column was properly identified")
        
        if stats['missing_amounts'] > 0:
            issues.append(f"{stats['missing_amounts']} transactions missing amounts")
            suggestions.append("Verify amount column and numeric parsing")
        
        if stats['zero_amounts'] > 0:
            issues.append(f"{stats['zero_amounts']} transactions have zero amount")
            suggestions.append("Review zero-amount transactions for validity")
        
        # Calculate quality score based on completeness
        quality_score = completeness
        
        if completeness < 100 and completeness >= 90:
            quality_score = 85  # Good but not perfect
        elif completeness < 90 and completeness >= 80:
            quality_score = 70  # Acceptable
        elif completeness < 80:
            quality_score = max(50, completeness - 10)  # Poor
        
        is_valid = quality_score >= self.min_quality_score
        
        return {
            "is_valid": is_valid,
            "quality_score": quality_score,
            "statistics": stats,
            "issues": issues,
            "suggestions": suggestions,
            "reflection": f"Rule-based validation: {completeness}% completeness",
            "needs_reextraction": not is_valid
        }
