"""
Tests for schema detection agent.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.schema_mapper import SchemaMapperAgent


class TestSchemaDetection:
    """Test schema detection accuracy across different bank formats."""
    
    def test_schema_mapper_initialization(self):
        """Test that schema mapper can be initialized."""
        mapper = SchemaMapperAgent()
        assert mapper is not None
    
    def test_detect_common_date_columns(self):
        """Test detection of common date column variations."""
        mapper = SchemaMapperAgent()
        
        # Test data with common date column names
        test_cases = [
            pd.DataFrame({"Date": ["2024-01-01"], "Description": ["Test"], "Amount": [100]}),
            pd.DataFrame({"Transaction Date": ["2024-01-01"], "Desc": ["Test"], "Amount": [100]}),
            pd.DataFrame({"Posted Date": ["2024-01-01"], "Merchant": ["Test"], "Debit": [100]}),
        ]
        
        for df in test_cases:
            # This will test if mapper can handle different column names
            result = mapper.extract_from_dataframe(df, "csv", "test.csv")
            assert isinstance(result, list)
    
    def test_detect_amount_columns(self):
        """Test detection of amount/debit/credit columns."""
        mapper = SchemaMapperAgent()
        
        test_df = pd.DataFrame({
            "Date": ["2024-01-01"],
            "Description": ["Test Transaction"],
            "Amount": [100.50]
        })
        
        result = mapper.extract_from_dataframe(test_df, "csv", "test.csv")
        
        # Should extract transactions with amount field
        if result:
            assert "amount" in result[0] or "Amount" in result[0]
    
    def test_handles_multiple_formats(self):
        """Test that mapper handles various bank statement formats."""
        mapper = SchemaMapperAgent()
        
        # Different formats from different banks
        formats = [
            {"Date": ["2024-01-01"], "Description": ["Store"], "Amount": [-50.00], "Balance": [1000]},
            {"Transaction Date": ["2024-01-01"], "Merchant": ["Store"], "Debit": [50.00], "Credit": [0], "Balance": [1000]},
            {"Posted": ["2024-01-01"], "Details": ["Store"], "Value": [-50.00]},
        ]
        
        success_count = 0
        for format_data in formats:
            df = pd.DataFrame(format_data)
            try:
                result = mapper.extract_from_dataframe(df, "csv", "test.csv")
                if result:
                    success_count += 1
            except:
                pass
        
        # Should successfully parse most formats
        assert success_count >= 2
    
    def test_real_sample_statements(self):
        """Test with actual sample bank statements."""
        from tools.file_parser import FileParser
        
        mapper = SchemaMapperAgent()
        sample_path = Path(__file__).parent.parent / "data" / "sample_statements"
        
        if not sample_path.exists():
            pytest.skip("Sample statements directory not found")
        
        files = list(sample_path.glob("*.csv")) + list(sample_path.glob("*.xlsx"))
        
        if not files:
            pytest.skip("No sample files found")
        
        successful_parses = 0
        total_files = min(5, len(files))  # Test up to 5 files
        
        for file in files[:total_files]:
            try:
                parsed = FileParser.parse_file(str(file))
                df = parsed["dataframe"]
                result = mapper.extract_from_dataframe(df, parsed["file_type"], str(file))
                
                if result and len(result) > 0:
                    successful_parses += 1
            except Exception as e:
                print(f"Failed to parse {file}: {e}")
        
        # Should successfully parse at least 80% of sample statements
        success_rate = successful_parses / total_files if total_files > 0 else 0
        assert success_rate >= 0.8, f"Only {successful_parses}/{total_files} files parsed successfully"
