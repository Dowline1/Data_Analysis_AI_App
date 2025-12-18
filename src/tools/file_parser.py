"""
File parsing tools for CSV, Excel, and PDF bank statements.
The PDF parser is designed to be flexible and work with different bank formats.
"""

import pandas as pd
import pdfplumber
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FileParser:
    """Handles parsing of different file formats into DataFrames."""
    
    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """
        Figure out what type of file we're dealing with.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File extension: 'csv', 'xlsx', or 'pdf'
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension in ['.csv', '.txt']:
            return 'csv'
        elif extension in ['.xlsx', '.xls']:
            return 'xlsx'
        elif extension == '.pdf':
            return 'pdf'
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    @staticmethod
    def parse_csv(file_path: str) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.
        Tries different delimiters and encodings in case the file is weird.
        """
        logger.info(f"Parsing CSV file: {file_path}")
        
        # Try common CSV formats
        try:
            # Standard comma-separated
            df = pd.read_csv(file_path)
            logger.info(f"Successfully parsed CSV with {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Standard CSV parse failed: {e}, trying semicolon separator")
            try:
                # Some European CSVs use semicolons
                df = pd.read_csv(file_path, sep=';')
                logger.info(f"Successfully parsed CSV with semicolon separator, {len(df)} rows")
                return df
            except Exception as e2:
                logger.error(f"Failed to parse CSV: {e2}")
                raise
    
    @staticmethod
    def parse_excel(file_path: str) -> pd.DataFrame:
        """Load Excel file into DataFrame."""
        logger.info(f"Parsing Excel file: {file_path}")
        
        try:
            # Read first sheet by default
            df = pd.read_excel(file_path)
            logger.info(f"Successfully parsed Excel with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to parse Excel: {e}")
            raise
    
    @staticmethod
    def parse_pdf(file_path: str) -> Tuple[pd.DataFrame, str]:
        """
        Extract tables from PDF bank statement.
        This is the tricky one - PDFs can be formatted in many different ways.
        
        Returns:
            Tuple of (DataFrame, raw_text) - the raw text is useful for debugging
        """
        logger.info(f"Parsing PDF file: {file_path}")
        
        try:
            all_tables = []
            raw_text = ""
            
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.debug(f"Processing page {page_num}")
                    
                    # Extract text for debugging/fallback
                    page_text = page.extract_text()
                    if page_text:
                        raw_text += page_text + "\n"
                    
                    # Try to extract tables from this page with more aggressive settings
                    # Many bank statements don't have clear table borders
                    table_settings = {
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 3,
                        "min_words_vertical": 3,
                        "min_words_horizontal": 1,
                    }
                    
                    # Try with strict settings first
                    tables = page.extract_tables(table_settings)
                    
                    # If no tables found, try with more lenient settings
                    if not tables:
                        table_settings = {
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "intersection_tolerance": 15,
                        }
                        tables = page.extract_tables(table_settings)
                    
                    if tables:
                        logger.info(f"Found {len(tables)} tables on page {page_num}")
                        
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 0:
                                # Convert table to DataFrame
                                # First row might be headers or might be data - we'll handle this later
                                # Make column names unique by appending index if duplicates exist
                                headers = table[0]
                                if headers:
                                    seen = {}
                                    unique_headers = []
                                    for col in headers:
                                        col_str = str(col) if col is not None else 'col'
                                        if col_str in seen:
                                            seen[col_str] += 1
                                            unique_headers.append(f"{col_str}_{seen[col_str]}")
                                        else:
                                            seen[col_str] = 0
                                            unique_headers.append(col_str)
                                    df_table = pd.DataFrame(table[1:], columns=unique_headers)
                                else:
                                    df_table = pd.DataFrame(table[1:])
                                
                                # Skip if table is too small (probably not transaction data)
                                if len(df_table) < 2:
                                    logger.debug(f"Skipping small table {table_idx} with {len(df_table)} rows")
                                    continue
                                
                                logger.info(f"Table {table_idx}: {len(df_table)} rows, {len(df_table.columns)} columns")
                                all_tables.append(df_table)
            
            if not all_tables:
                logger.warning("No tables found in PDF, will need to parse raw text")
                # If no tables found, return empty DataFrame with raw text for LLM to parse
                return pd.DataFrame(), raw_text
            
            # Combine all tables (in case transactions span multiple pages)
            if len(all_tables) == 1:
                combined_df = all_tables[0]
            else:
                logger.info(f"Combining {len(all_tables)} tables")
                # Tables from different pages might have different column counts
                # Find the table with the most columns as it likely has complete headers
                max_cols = max(len(df.columns) for df in all_tables)
                logger.info(f"Table column counts: {[len(df.columns) for df in all_tables]}, using {max_cols} columns")
                
                # Pad smaller tables with empty columns
                padded_tables = []
                for df in all_tables:
                    if len(df.columns) < max_cols:
                        # Add empty columns to match max
                        for i in range(len(df.columns), max_cols):
                            df[f'col_{i}'] = None
                    padded_tables.append(df)
                
                combined_df = pd.concat(padded_tables, ignore_index=True)
            
            # Clean up the DataFrame
            # Remove completely empty rows
            combined_df = combined_df.dropna(how='all')
            
            # Remove duplicate header rows that might appear on each page
            # (e.g., if "Date" appears multiple times, it's probably a repeated header)
            if len(combined_df) > 0:
                first_row = combined_df.iloc[0].astype(str).str.lower()
                duplicate_headers = combined_df[combined_df.astype(str).apply(lambda x: x.str.lower()).eq(first_row).all(axis=1)]
                if len(duplicate_headers) > 1:
                    logger.info(f"Removing {len(duplicate_headers) - 1} duplicate header rows")
                    combined_df = combined_df[~combined_df.index.isin(duplicate_headers.index[1:])]
            
            logger.info(f"Final parsed DataFrame: {len(combined_df)} rows, {len(combined_df.columns)} columns")
            logger.debug(f"Columns found: {combined_df.columns.tolist()}")
            
            return combined_df, raw_text
            
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise
    
    @classmethod
    def parse_file(cls, file_path: str) -> Dict:
        """
        Main entry point - parse any supported file type.
        
        Args:
            file_path: Path to the bank statement file
            
        Returns:
            Dict containing:
                - 'dataframe': Parsed pandas DataFrame
                - 'raw_text': Raw extracted text (for PDFs)
                - 'file_type': Type of file parsed
                - 'columns': List of column names found
        """
        file_type = cls.detect_file_type(file_path)
        logger.info(f"Detected file type: {file_type}")
        
        raw_text = None
        
        if file_type == 'csv':
            df = cls.parse_csv(file_path)
        elif file_type == 'xlsx':
            df = cls.parse_excel(file_path)
        elif file_type == 'pdf':
            df, raw_text = cls.parse_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return {
            'dataframe': df,
            'raw_text': raw_text,
            'file_type': file_type,
            'columns': df.columns.tolist() if df is not None and not df.empty else []
        }
