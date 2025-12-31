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
    account_type: Optional[str] = Field(default=None, description="Account type: checking, savings, credit_card, etc.")


class TransactionsOutput(BaseModel):
    """List of extracted transactions."""
    transactions: List[TransactionSchema] = Field(description="List of all transactions found")


class SchemaMapperAgent:
    """
    ReAct agent that intelligently extracts transactions from any bank statement format.
    Works with CSVs or Excel files regardless of their column names or structure.
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
- CSV or Excel files (from any bank)
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
    
    def extract_from_dataframe(self, df: pd.DataFrame, file_type: str, file_path: str = None) -> List[Dict]:
        """
        Extract transactions from a DataFrame using smart column mapping.
        Uses LLM once to identify columns, then Python to extract all rows.
        
        Args:
            df: Pandas DataFrame with bank statement data
            file_type: Type of file ('csv' or 'xlsx')
            file_path: Optional file path to help detect account type
            
        Returns:
            List of transaction dictionaries
        """
        logger.info(f"Extracting transactions from {file_type} DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        # Step 1: Detect account type from file name and content
        account_type = self._detect_account_type(file_path, df)
        logger.info(f"Detected account type: {account_type}")
        
        # Step 2: Use LLM once to map columns (fast!)
        column_mapping = self._identify_columns(df)
        
        if not column_mapping:
            logger.warning("Failed to identify columns, falling back to full LLM extraction")
            # Fallback to old method for small dataframes
            if len(df) <= 100:
                columns_str = ", ".join(df.columns.tolist())
                raw_data = df.to_string()
                return self._extract_with_llm(file_type, columns_str, raw_data)
            else:
                logger.error("Cannot process large DataFrame without column mapping")
                return []
        
        # Step 3: Use pure Python to extract all transactions (instant!)
        logger.info(f"Using column mapping to extract {len(df)} transactions")
        transactions = self._extract_with_mapping(df, column_mapping, account_type)
        
        logger.info(f"Extracted {len(transactions)} transactions using smart mapping")
        return transactions
    
    def _classify_account_type(self, account_name: str) -> str:
        """
        Classify account type from account name/label.
        
        Args:
            account_name: Account name like "Credit Card *1234", "Checking", "Current Account"
            
        Returns:
            Normalized account type: 'credit_card', 'checking', 'savings', 'current', or original name
        """
        account_lower = account_name.lower()
        
        # Check for common patterns
        if any(keyword in account_lower for keyword in ['credit card', 'visa', 'mastercard', 'amex', 'cc ']):
            return 'credit_card'
        elif any(keyword in account_lower for keyword in ['checking', 'chequing']):
            return 'checking'
        elif any(keyword in account_lower for keyword in ['current account', 'current']):
            return 'current'
        elif 'saving' in account_lower:
            return 'savings'
        else:
            # Return a cleaned version of the account name
            return account_name.replace(' ', '_').lower()
    
    def _identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Use LLM once to identify which columns contain transaction data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict mapping field names to column names: {'date': 'Transaction Date', 'amount': 'Amount', ...}
        """
        logger.info("Identifying column mapping using LLM")
        
        # Skip potential header/footer rows that may have empty or non-transaction data
        # Find rows that look like actual transaction data (have non-empty values in multiple columns)
        # Count non-empty string values (not just non-null)
        def count_non_empty(row):
            return sum(1 for val in row if pd.notna(val) and str(val).strip() != '')
        
        non_empty_counts = df.apply(count_non_empty, axis=1)
        potential_data_rows = df[non_empty_counts >= 3]  # Rows with at least 3 non-empty values
        
        if len(potential_data_rows) == 0:
            logger.warning("No rows with sufficient non-empty values found")
            return None
            
        # Show LLM data from different parts of the dataset
        logger.info(f"Found {len(potential_data_rows)} potential data rows out of {len(df)} total")
        
        # Take samples from beginning, middle, and end of potential data rows
        sample_indices = []
        if len(potential_data_rows) > 0:
            sample_indices.append(0)
        if len(potential_data_rows) > 10:
            sample_indices.append(len(potential_data_rows) // 2)
        if len(potential_data_rows) > 20:
            sample_indices.append(len(potential_data_rows) - 1)
        
        sample_rows = potential_data_rows.iloc[sample_indices] if sample_indices else potential_data_rows.head(5)
        
        # Format column names for display, handling empty/unnamed columns
        col_display = []
        for i, col in enumerate(df.columns):
            if col == '' or col is None or str(col).strip() == '':
                col_display.append(f"Column_{i} (unnamed)")
            else:
                col_display.append(col)
        
        columns_info = "Here are the column names and sample transaction data from a bank statement.\n"
        columns_info += f"Total columns: {len(df.columns)}\n\n"
        columns_info += f"Column names: {', '.join(col_display)}\n\n"
        columns_info += f"Sample transaction rows (showing only non-empty fields):\n"
        
        # Show each sample row with only non-empty columns
        for i, idx in enumerate(sample_rows.index[:5], 1):  # Show up to 5 samples
            row = df.loc[idx]
            columns_info += f"\nTransaction {i}:\n"
            for j, col in enumerate(df.columns):
                val = row[col]
                if pd.notna(val) and str(val).strip() != '':
                    display_name = col_display[j]
                    columns_info += f"  {display_name}: {val!r}\n"
        
        from langchain_core.output_parsers import JsonOutputParser
        from pydantic import BaseModel, Field
        from langchain_core.prompts import ChatPromptTemplate
        
        class ColumnMapping(BaseModel):
            date_column: str = Field(description="Name of column containing transaction dates")
            description_column: str = Field(description="Name of column containing transaction descriptions/merchants")
            amount_column: str = Field(description="Name of column containing transaction amounts")
            debit_column: str | None = Field(default=None, description="Name of column for debits (if separate from amount)")
            credit_column: str | None = Field(default=None, description="Name of column for credits (if separate from amount)")
            category_column: str | None = Field(default=None, description="Name of column for transaction category/type (e.g., 'Transfer', 'Card Payment', 'Groceries')")
            type_column: str | None = Field(default=None, description="Name of column for transaction type (e.g., 'Debit', 'Credit', 'ATM')")
            fee_column: str | None = Field(default=None, description="Name of column for transaction fees")
            currency_column: str | None = Field(default=None, description="Name of column for currency")
            balance_column: str | None = Field(default=None, description="Name of column for running balance after transaction")
            status_column: str | None = Field(default=None, description="Name of column for transaction status (e.g., 'COMPLETED', 'PENDING')")
            account_name_column: str | None = Field(default=None, description="Name of column for account name/number (e.g., 'Account', 'Card Number', 'Account Name')")
        
        parser = JsonOutputParser(pydantic_object=ColumnMapping)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a bank statement table to identify ALL relevant columns.

Look at the sample transaction data and identify these columns (REQUIRED fields first):

REQUIRED:
1. DATE column: Transaction dates ("1 Nov 2025", "2024-11-01", etc.)
2. DESCRIPTION column: Merchant names or transaction descriptions ("Tesco", "Levi's", etc.)
3. AMOUNT column: Monetary amounts ("€18.02", "-50.00", "100.50", etc.)
   - OR separate DEBIT and CREDIT columns if amounts are split

OPTIONAL (identify if present):
4. CATEGORY column: Transaction category/type ("Transfer", "Card Payment", "Groceries", "Shopping", etc.)
5. TYPE column: Transaction method/type ("Debit", "Credit", "ATM", "Online", etc.)
6. FEE column: Transaction fees or charges
7. CURRENCY column: Currency code ("EUR", "USD", "GBP", etc.)
8. BALANCE column: Running account balance after transaction
9. STATUS column: Transaction status ("COMPLETED", "PENDING", "FAILED", etc.)
10. ACCOUNT_NAME column: Account name/number ("Checking *1234", "Visa *5678", "Current Account", etc.)

Return the EXACT column names as they appear in the column list.
For optional fields, return null if not present.
If a column is named "Column_X (unnamed)", use that exact string.

{format_instructions}"""),
            ("user", "{columns_info}")
        ])
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "columns_info": columns_info,
                "format_instructions": parser.get_format_instructions()
            })
            
            logger.info(f"Raw LLM column mapping: {result}")
            
            # Map display names back to actual column names
            # Handle "Column_X (unnamed)" format
            def map_to_actual_column(display_name):
                if display_name and "Column_" in display_name and "(unnamed)" in display_name:
                    # Extract index from "Column_X (unnamed)"
                    try:
                        idx = int(display_name.split("_")[1].split(" ")[0])
                        return df.columns[idx]
                    except:
                        pass
                return display_name
            
            mapped_result = {
                'date_column': map_to_actual_column(result.get('date_column')),
                'description_column': map_to_actual_column(result.get('description_column')),
                'amount_column': map_to_actual_column(result.get('amount_column')),
                'debit_column': map_to_actual_column(result.get('debit_column')) if result.get('debit_column') else None,
                'credit_column': map_to_actual_column(result.get('credit_column')) if result.get('credit_column') else None,
                'category_column': map_to_actual_column(result.get('category_column')) if result.get('category_column') else None,
                'type_column': map_to_actual_column(result.get('type_column')) if result.get('type_column') else None,
                'fee_column': map_to_actual_column(result.get('fee_column')) if result.get('fee_column') else None,
                'currency_column': map_to_actual_column(result.get('currency_column')) if result.get('currency_column') else None,
                'balance_column': map_to_actual_column(result.get('balance_column')) if result.get('balance_column') else None,
                'status_column': map_to_actual_column(result.get('status_column')) if result.get('status_column') else None,
                'account_name_column': map_to_actual_column(result.get('account_name_column')) if result.get('account_name_column') else None,
            }
            
            logger.info(f"Mapped to actual columns: {mapped_result}")
            logger.info(f"Optional metadata columns found: category={mapped_result.get('category_column')}, type={mapped_result.get('type_column')}, fee={mapped_result.get('fee_column')}, currency={mapped_result.get('currency_column')}, balance={mapped_result.get('balance_column')}, status={mapped_result.get('status_column')}")
            return mapped_result
            
        except Exception as e:
            logger.error(f"Failed to identify columns: {e}")
            return None
    
    def _detect_account_type(self, file_path: str, df: pd.DataFrame) -> str:
        """
        Detect account type from multiple sources: file name, column data, and transaction patterns.
        This works dynamically regardless of bank format.
        
        Args:
            file_path: Path to the file (may contain hints like 'credit_card', 'checking')
            df: DataFrame to analyze for account type hints
            
        Returns:
            Account type string: 'checking', 'savings', 'credit_card', 'current', 'unknown'
        """
        import os
        
        logger.info("Detecting account type from multiple sources...")
        
        # Strategy 1: Check file name for hints
        if file_path:
            file_name = os.path.basename(file_path).lower()
            
            if any(keyword in file_name for keyword in ['credit', 'cc', 'visa', 'mastercard', 'amex', 'card']):
                logger.info(f"Account type detected from filename: credit_card")
                return 'credit_card'
            elif any(keyword in file_name for keyword in ['checking', 'chequing', 'current']):
                logger.info(f"Account type detected from filename: checking")
                return 'checking'
            elif 'saving' in file_name:
                logger.info(f"Account type detected from filename: savings")
                return 'savings'
        
        # Strategy 2: Check if there's an account name/type column
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['account', 'card', 'product']):
                # Check a few values to see if they indicate account type
                sample_values = df[col].dropna().head(10).astype(str).str.lower()
                for val in sample_values:
                    if any(keyword in val for keyword in ['credit', 'visa', 'mastercard', 'amex', 'card']):
                        logger.info(f"Account type detected from column '{col}': credit_card")
                        return 'credit_card'
                    elif any(keyword in val for keyword in ['checking', 'current', 'chequing']):
                        logger.info(f"Account type detected from column '{col}': checking")
                        return 'checking'
                    elif 'saving' in val:
                        logger.info(f"Account type detected from column '{col}': savings")
                        return 'savings'
        
        # Strategy 3: Use LLM to analyze transaction patterns and descriptions
        logger.info("Using LLM to analyze transaction patterns for account type...")
        sample_rows = df.head(15).to_string(max_colwidth=50)
        
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze this bank statement data and determine the account type.

Account types to identify:
- credit_card: Credit card statement with purchases/payments
- checking/current: Regular checking or current account  
- savings: Savings account
- unknown: Cannot determine

Look for these clues:
1. Transaction descriptions (e.g., "CREDIT CARD PAYMENT", "ATM WITHDRAWAL", "DEBIT CARD PURCHASE")
2. Account names in columns (e.g., "Visa *1234", "Checking *5678")
3. Balance patterns (credit cards often show purchases increasing balance)
4. Transaction types common to specific accounts

IMPORTANT: Different banks format data differently:
- Some show credit card purchases as positive, others as negative
- Column names vary widely
- Look at the CONTENT and PATTERNS, not just column names

Return ONLY one of these exact strings: credit_card, checking, savings, unknown"""),
            ("user", "Sample data from statement (max 15 rows):\n\n{sample_rows}\n\nWhat type of account is this?")
        ])
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"sample_rows": sample_rows})
            account_type = result.strip().lower()
            
            if account_type in ['credit_card', 'checking', 'savings', 'current']:
                logger.info(f"Account type detected by LLM: {account_type}")
                return account_type
            else:
                logger.warning(f"LLM returned unexpected account type: {result}, defaulting to unknown")
                return 'unknown'
        except Exception as e:
            logger.error(f"Failed to detect account type with LLM: {e}")
            return 'unknown'
    
    def _extract_with_mapping(self, df: pd.DataFrame, mapping: Dict, account_type: str = 'unknown') -> List[Dict]:
        """
        Extract transactions using identified column mapping (pure Python - fast!).
        
        Args:
            df: DataFrame with transaction data
            mapping: Column mapping from _identify_columns
            account_type: Detected account type for proper sign handling
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        
        date_col = mapping.get('date_column')
        desc_col = mapping.get('description_column')
        amount_col = mapping.get('amount_column')
        debit_col = mapping.get('debit_column')
        credit_col = mapping.get('credit_column')
        
        logger.info(f"Extracting using mapping: date={date_col}, desc={desc_col}, amount={amount_col}")
        
        # Filter to rows that have non-empty values in the key columns
        # This skips header/footer rows
        df_filtered = df[
            df[date_col].notna() & 
            (df[date_col].astype(str).str.strip() != '') &
            df[desc_col].notna() &
            (df[desc_col].astype(str).str.strip() != '') &
            df[amount_col].notna() &
            (df[amount_col].astype(str).str.strip() != '')
        ].copy()
        
        logger.info(f"Filtered to {len(df_filtered)} rows with non-empty values in key columns")
        
        for idx, row in df_filtered.iterrows():
            try:
                # Get date
                date_val = row.get(date_col)
                if pd.isna(date_val) or date_val is None or str(date_val).strip() == '':
                    continue
                
                # Parse date
                date_str = str(date_val).strip()
                if not date_str or date_str.lower() == 'nan' or date_str.lower() == 'none':
                    continue
                    
                date_obj = pd.to_datetime(date_str, errors='coerce')
                
                if pd.isna(date_obj):
                    continue
                
                # Get description
                description = str(row.get(desc_col, '')).strip()
                if not description or description.lower() in ['nan', 'none', '']:
                    continue
                
                # Get amount
                if debit_col and credit_col:
                    # Separate debit/credit columns
                    debit = pd.to_numeric(row.get(debit_col, 0), errors='coerce')
                    credit = pd.to_numeric(row.get(credit_col, 0), errors='coerce')
                    
                    if not pd.isna(debit) and debit != 0:
                        amount = -abs(float(debit))  # Debits are negative
                    elif not pd.isna(credit) and credit != 0:
                        amount = abs(float(credit))  # Credits are positive
                    else:
                        continue
                else:
                    # Single amount column
                    amount_val = row.get(amount_col)
                    amount = pd.to_numeric(amount_val, errors='coerce')
                    
                    if pd.isna(amount) or amount == 0:
                        continue
                    
                    amount = float(amount)
                
                # Determine account type for this transaction FIRST
                # Priority: 1) Account name column, 2) File-level detection
                transaction_account_type = account_type  # Default to file-level
                account_name_str = None
                account_name_col = mapping.get('account_name_column')
                
                if account_name_col and account_name_col in df.columns:
                    account_name = row.get(account_name_col)
                    if pd.notna(account_name) and str(account_name).strip():
                        account_name_str = str(account_name).strip()
                        # Classify account type from account name
                        transaction_account_type = self._classify_account_type(account_name_str)
                
                # Get transaction type from column if available
                type_col = mapping.get('type_column')
                transaction_type = None
                if type_col and type_col in df.columns:
                    type_val = row.get(type_col)
                    if pd.notna(type_val) and str(type_val).strip():
                        transaction_type = str(type_val).strip().lower()
                
                # Apply sign conventions dynamically based on BOTH account type AND transaction type
                # This ensures accurate balance calculations for each account type
                
                if transaction_account_type == 'credit_card':
                    # CREDIT CARDS: Track debt
                    # - Debit/Charge = Purchase = POSITIVE (increases debt owed)
                    # - Credit/Payment = Payment = NEGATIVE (reduces debt owed)
                    if transaction_type:
                        if transaction_type in ['debit', 'charge', 'purchase', 'dr']:
                            amount = abs(amount)  # Purchase increases debt
                        elif transaction_type in ['credit', 'payment', 'cr', 'refund']:
                            amount = -abs(amount)  # Payment reduces debt
                        else:
                            # Fallback: use description
                            if 'payment' in description.lower() or 'paid' in description.lower():
                                amount = -abs(amount)
                            else:
                                amount = abs(amount)
                    else:
                        # No transaction type, use description
                        if 'payment' in description.lower() or 'paid' in description.lower():
                            amount = -abs(amount)  # Payment
                        else:
                            amount = abs(amount)  # Purchase
                            
                else:
                    # CHECKING/SAVINGS/CURRENT ACCOUNTS: Track available funds
                    # - Debit/Withdrawal = Money out = NEGATIVE (reduces balance)
                    # - Credit/Deposit = Money in = POSITIVE (increases balance)
                    if transaction_type:
                        if transaction_type in ['debit', 'withdrawal', 'payment', 'spent', 'out', 'dr']:
                            amount = -abs(amount)  # Money out
                        elif transaction_type in ['credit', 'deposit', 'income', 'received', 'in', 'cr']:
                            amount = abs(amount)  # Money in
                        # else: keep original sign from amount column
                    # else: keep original sign from amount column
                        
                # Create transaction with all available metadata
                transaction = {
                    'date': date_obj.to_pydatetime(),
                    'description': description,
                    'amount': amount,
                    'category': None,  # Will be enriched below
                    'account_type': transaction_account_type,  # Per-transaction account type
                    'account_name': account_name_str
                }
                
                # Add optional metadata if columns are present
                category_col = mapping.get('category_column')
                if category_col and category_col in df.columns:
                    cat_val = row.get(category_col)
                    if pd.notna(cat_val) and str(cat_val).strip():
                        transaction['category'] = str(cat_val).strip()
                
                if transaction_type:
                    transaction['type'] = transaction_type
                
                fee_col = mapping.get('fee_column')
                if fee_col and fee_col in df.columns:
                    fee_val = pd.to_numeric(row.get(fee_col), errors='coerce')
                    if pd.notna(fee_val):
                        transaction['fee'] = float(fee_val)
                
                currency_col = mapping.get('currency_column')
                if currency_col and currency_col in df.columns:
                    curr_val = row.get(currency_col)
                    if pd.notna(curr_val) and str(curr_val).strip():
                        transaction['currency'] = str(curr_val).strip()
                
                balance_col = mapping.get('balance_column')
                if balance_col and balance_col in df.columns:
                    bal_val = pd.to_numeric(row.get(balance_col), errors='coerce')
                    if pd.notna(bal_val):
                        transaction['balance'] = float(bal_val)
                
                status_col = mapping.get('status_column')
                if status_col and status_col in df.columns:
                    status_val = row.get(status_col)
                    if pd.notna(status_val) and str(status_val).strip():
                        transaction['status'] = str(status_val).strip()
                
                transactions.append(transaction)
                
            except Exception as e:
                logger.debug(f"Skipping row {idx}: {e}")
                continue
        
        return transactions
    
    def extract_from_text(self, text: str, file_type: str = 'csv') -> List[Dict]:
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
                - 'raw_text': raw text (if available)
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
