"""
Tests for file parsing functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools.file_parser import FileParser


class TestFileParser:
    """Test file parsing logic for CSV and Excel files."""
    
    def test_parse_csv_file(self):
        """Test parsing a CSV file."""
        # Test with sample data
        sample_path = Path(__file__).parent.parent / "data" / "sample_statements"
        
        # Find any CSV file in samples
        csv_files = list(sample_path.glob("*.csv"))
        if csv_files:
            result = FileParser.parse_file(str(csv_files[0]))
            
            assert result is not None
            assert "dataframe" in result
            assert "file_type" in result
            assert isinstance(result["dataframe"], pd.DataFrame)
            assert result["file_type"] == "csv"
            assert len(result["dataframe"]) > 0
    
    def test_parse_excel_file(self):
        """Test parsing an Excel file."""
        sample_path = Path(__file__).parent.parent / "data" / "sample_statements"
        
        # Find any Excel file in samples
        excel_files = list(sample_path.glob("*.xlsx")) + list(sample_path.glob("*.xls"))
        if excel_files:
            result = FileParser.parse_file(str(excel_files[0]))
            
            assert result is not None
            assert "dataframe" in result
            assert "file_type" in result
            assert isinstance(result["dataframe"], pd.DataFrame)
            assert result["file_type"] in ["xlsx", "xls"]
            assert len(result["dataframe"]) > 0
    
    def test_invalid_file_extension(self):
        """Test that invalid file extensions are handled."""
        with pytest.raises(Exception):
            FileParser.parse_file("test.txt")
    
    def test_nonexistent_file(self):
        """Test that nonexistent files raise an error."""
        with pytest.raises(Exception):
            FileParser.parse_file("nonexistent_file.csv")
    
    def test_dataframe_has_columns(self):
        """Test that parsed file has column data."""
        sample_path = Path(__file__).parent.parent / "data" / "sample_statements"
        
        # Get first available file
        files = list(sample_path.glob("*.csv")) + list(sample_path.glob("*.xlsx"))
        if files:
            result = FileParser.parse_file(str(files[0]))
            df = result["dataframe"]
            
            assert len(df.columns) > 0
            assert len(df) > 0
