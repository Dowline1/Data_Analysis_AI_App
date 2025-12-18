"""
Schema Mapper Agent - Uses LLM to intelligently map any bank statement format to our standard schema.
This is a ReAct agent that can understand different bank formats and extract transactions.
"""

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TransactionSchema(BaseModel):
    """Expected output format for a single transaction."""
    date: str = Field(description="Transaction date in YYYY-MM-DD format")
    description: str = Field(description="Transaction description or merchant name")
    amount: float = Field(description="Transaction amount (positive for money in, negative for money out)")
    category: Optional[str] = Field(default=None, description="Transaction category if identifiable")


class TransactionsOutput(BaseModel):
    """List of extracted transactions."""
    transactions: List[TransactionSchema] = Field(description="List of all transactions found")


class SchemaMapperAgent:
    """
    ReAct agent that intelligently extracts transactions from any bank statement format.
    Works with PDFs, CSVs, or Excel files regardless of their column names or structure.
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=Config.GEMINI_TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.parser = JsonOutputParser(pydantic_object=TransactionsOutput)
        
        # Create the prompt template
        # This prompt guides the LLM to extract transactions regardless of format
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at reading bank statements and extracting transaction data.
Your job is to analyze the provided bank statement data and extract ALL transactions.

Bank statements can come in many formats:
- PDF text (like Revolut, AIB, Bank of Ireland)
- CSV files with various column names
- Excel spreadsheets

Look for these key pieces of information for each transaction:
1. Date (could be labeled as: Date, Transaction Date, Posted Date, Value Date, etc.)
2. Description (could be: Description, Details, Merchant, Narrative, etc.)
3. Amount (could be: Amount, Debit, Credit, Value, Balance Change, etc.)
   - If there are separate Debit/Credit columns, combine them (debits as negative, credits as positive)
   - If amounts are always positive, look for context to determine if it's money in or out
4. Category (optional - only if explicitly provided in the statement)

IMPORTANT RULES:
- Extract EVERY transaction you can find
- Skip header rows, summary rows, and account information
- Convert all dates to YYYY-MM-DD format
- Make amounts negative for money out (purchases, withdrawals) and positive for money in (deposits, refunds)
- Keep descriptions concise but informative
- If you can't determine something with confidence, use null for optional fields

{format_instructions}"""),
            ("user", """Here is the bank statement data to analyze:

File Type: {file_type}
Columns Found: {columns}

Raw Data:
{raw_data}

Please extract all transactions from this statement.""")
        ])
    
    def extract_from_dataframe(self, df: pd.DataFrame, file_type: str) -> List[Dict]:
        """
        Extract transactions from a DataFrame (CSV or Excel).
        Processes in batches for large DataFrames.
        
        Args:
            df: Pandas DataFrame with bank statement data
            file_type: Type of file ('csv' or 'xlsx')
            
        Returns:
            List of transaction dictionaries
        """
        logger.info(f"Extracting transactions from {file_type} DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        columns_str = ", ".join(df.columns.tolist())
        
        # For small dataframes, process all at once
        if len(df) <= 100:
            raw_data = df.to_string()
            return self._extract_with_llm(file_type, columns_str, raw_data)
        
        # For large dataframes, process in batches
        logger.info(f"Large DataFrame detected, processing in batches")
        batch_size = 100
        all_transactions = []
        
        for batch_num in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_num:batch_num + batch_size]
            batch_end = min(batch_num + batch_size, len(df))
            
            logger.info(f"Processing rows {batch_num + 1} to {batch_end}")
            
            # Include headers in each batch for context
            raw_data = f"Rows {batch_num + 1} to {batch_end}:\n{batch_df.to_string()}"
            
            batch_transactions = self._extract_with_llm(file_type, columns_str, raw_data)
            
            if batch_transactions:
                logger.info(f"Extracted {len(batch_transactions)} transactions from batch")
                all_transactions.extend(batch_transactions)
        
        logger.info(f"Total extracted: {len(all_transactions)} transactions from all batches")
        return all_transactions
    
    def extract_from_text(self, text: str, file_type: str = 'pdf') -> List[Dict]:
        """
        Extract transactions from raw text (usually from PDF).
        Uses chunking for large documents to extract all transactions.
        
        Args:
            text: Raw text extracted from bank statement
            file_type: Type of file (default 'pdf')
            
        Returns:
            List of transaction dictionaries
        """
        logger.info(f"Extracting transactions from {file_type} text ({len(text)} chars)")
        
        # Gemini has 1M token context, but let's chunk large documents for efficiency
        # ~4 chars per token, so 32k chars = ~8k tokens (safe for most models)
        max_chunk_size = 32000
        
        if len(text) <= max_chunk_size:
            # Small enough to process in one go
            return self._extract_with_llm(file_type, "N/A (text-based)", text)
        
        # For large documents, split into chunks and extract from each
        logger.info(f"Text is {len(text)} chars, splitting into chunks for complete extraction")
        
        all_transactions = []
        chunk_size = max_chunk_size
        overlap = 500  # Overlap to avoid cutting transactions
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunk_num = (i // (chunk_size - overlap)) + 1
            total_chunks = (len(text) + chunk_size - overlap - 1) // (chunk_size - overlap)
            
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} chars)")
            
            chunk_transactions = self._extract_with_llm(file_type, "N/A (text-based)", chunk)
            
            if chunk_transactions:
                logger.info(f"Extracted {len(chunk_transactions)} transactions from chunk {chunk_num}")
                all_transactions.extend(chunk_transactions)
        
        # Remove duplicates that might appear in overlapping sections
        # Deduplicate by date+description+amount
        seen = set()
        unique_transactions = []
        
        for txn in all_transactions:
            # Create a unique key from transaction details
            key = (str(txn.get('date')), txn.get('description', ''), txn.get('amount', 0))
            if key not in seen:
                seen.add(key)
                unique_transactions.append(txn)
        
        if len(all_transactions) != len(unique_transactions):
            logger.info(f"Removed {len(all_transactions) - len(unique_transactions)} duplicate transactions from overlapping chunks")
        
        return unique_transactions
    
    def _extract_with_llm(self, file_type: str, columns: str, raw_data: str) -> List[Dict]:
        """
        Use LLM to extract transactions from the data.
        
        Args:
            file_type: Type of source file
            columns: Column names (or "N/A" for text)
            raw_data: The actual data to parse
            
        Returns:
            List of transaction dictionaries
        """
        try:
            # Create the chain
            chain = self.prompt | self.llm | self.parser
            
            # Run the extraction
            logger.info("Calling LLM to extract transactions...")
            result = chain.invoke({
                "file_type": file_type,
                "columns": columns,
                "raw_data": raw_data,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Extract transactions from result
            # Result is already a dict from Pydantic parser
            if isinstance(result, dict):
                transactions = result.get('transactions', [])
            else:
                # If result is the TransactionsOutput object directly
                transactions = result.transactions if hasattr(result, 'transactions') else []
            
            logger.info(f"Successfully extracted {len(transactions)} transactions")
            
            # Convert to list of dicts
            transactions_list = []
            for t in transactions:
                # Parse date if it's a string
                try:
                    date_obj = datetime.strptime(t['date'], '%Y-%m-%d')
                except:
                    logger.warning(f"Could not parse date: {t['date']}, using current date")
                    date_obj = datetime.now()
                
                transactions_list.append({
                    'date': date_obj,
                    'description': t['description'],
                    'amount': float(t['amount']),
                    'category': t.get('category')
                })
            
            return transactions_list
            
        except Exception as e:
            logger.error(f"Error extracting transactions with LLM: {e}")
            raise
    
    def map_statement(self, parsed_data: Dict) -> List[Dict]:
        """
        Main entry point - map any bank statement to standard transaction format.
        
        Args:
            parsed_data: Dictionary containing:
                - 'dataframe': pandas DataFrame (if CSV/Excel)
                - 'raw_text': raw text (if PDF)
                - 'file_type': type of file
                
        Returns:
            List of standardized transaction dictionaries
        """
        file_type = parsed_data['file_type']
        df = parsed_data.get('dataframe')
        raw_text = parsed_data.get('raw_text')
        
        # Decide which extraction method to use
        if df is not None and not df.empty:
            logger.info(f"Using DataFrame extraction for {file_type}")
            return self.extract_from_dataframe(df, file_type)
        elif raw_text:
            logger.info(f"Using text extraction for {file_type}")
            return self.extract_from_text(raw_text, file_type)
        else:
            raise ValueError("No usable data found in parsed_data")
