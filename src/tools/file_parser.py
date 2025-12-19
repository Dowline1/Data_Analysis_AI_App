"""
File parsing tools for CSV and Excel bank statements.
"""

import pandas as pd
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
            File extension: 'csv' or 'xlsx'
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension in ['.csv', '.txt']:
            return 'csv'
        elif extension in ['.xlsx', '.xls']:
            return 'xlsx'
        else:
            raise ValueError(f"Unsupported file type: {extension}. Please use CSV or Excel files.")
    
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
    
    @classmethod
    def parse_file(cls, file_path: str) -> Dict:
        """
        Main entry point - parse any supported file type.
        
        Args:
            file_path: Path to the bank statement file
            
        Returns:
            Dict containing:
                - 'dataframe': Parsed pandas DataFrame
                - 'file_type': Type of file parsed
                - 'columns': List of column names found
        """
        file_type = cls.detect_file_type(file_path)
        logger.info(f"Detected file type: {file_type}")
        
        if file_type == 'csv':
            df = cls.parse_csv(file_path)
        elif file_type == 'xlsx':
            df = cls.parse_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return {
            'dataframe': df,
            'file_type': file_type,
            'columns': df.columns.tolist() if df is not None and not df.empty else []
        }
