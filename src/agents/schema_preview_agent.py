"""
Schema Preview Agent - HITL Schema/Account Confirmation
Provides a lightweight preview of detected schema and accounts for human validation.
"""

from typing import Dict, List, Optional
import pandas as pd
from src.agents.schema_mapper import SchemaMapperAgent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SchemaPreviewAgent:
    """
    Generates a preview of detected schema and accounts for HITL confirmation.
    This is a lightweight operation that doesn't run the full analysis.
    """
    
    def __init__(self):
        self.mapper = SchemaMapperAgent()
    
    def generate_preview(self, df: pd.DataFrame, file_type: str, filename: str = "") -> Dict:
        """
        Generate a preview of detected schema and accounts without running full extraction.
        
        Args:
            df: Parsed dataframe from file
            file_type: Type of file (csv, xlsx)
            filename: Original filename
            
        Returns:
            Dict containing:
                - detected_columns: Mapping of standard fields to detected columns
                - account_preview: List of detected accounts with sample data
                - sample_data: First few rows for preview
                - recommendations: Suggested corrections/warnings
        """
        logger.info(f"Generating schema preview for {filename}")
        
        # Detect column mappings using existing logic
        column_mapping = self.mapper._identify_columns(df)
        
        # Detect accounts from sample data
        account_preview = self._preview_accounts(df, column_mapping)
        
        # Get sample data
        sample_data = self._get_sample_data(df, column_mapping)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(column_mapping, account_preview, df)
        
        return {
            'detected_columns': column_mapping,
            'account_preview': account_preview,
            'sample_data': sample_data,
            'recommendations': recommendations,
            'total_rows': len(df)
        }
    
    def _preview_accounts(self, df: pd.DataFrame, column_mapping: Dict) -> List[Dict]:
        """
        Preview detected accounts from the account name column.
        
        Returns:
            List of dicts with account info:
                - account_name: Original name from file
                - account_type: Detected type (credit_card, checking, etc.)
                - transaction_count: Number of transactions
                - sample_transactions: First 3 transaction descriptions
        """
        account_col = column_mapping.get('account_name_column')
        if not account_col or account_col not in df.columns:
            return []
        
        accounts = []
        for account_name in df[account_col].dropna().unique():
            account_type = self.mapper._classify_account_type(account_name)
            
            # Get transactions for this account
            account_txs = df[df[account_col] == account_name]
            
            # Get sample descriptions
            desc_col = column_mapping.get('description_column')
            sample_descs = []
            if desc_col and desc_col in df.columns:
                sample_descs = account_txs[desc_col].dropna().head(3).tolist()
            
            accounts.append({
                'account_name': account_name,
                'account_type': account_type,
                'transaction_count': len(account_txs),
                'sample_transactions': sample_descs
            })
        
        return sorted(accounts, key=lambda x: x['transaction_count'], reverse=True)
    
    def _get_sample_data(self, df: pd.DataFrame, column_mapping: Dict) -> pd.DataFrame:
        """Get first 10 rows with relevant columns."""
        relevant_cols = [
            col for col in [
                column_mapping.get('date_column'),
                column_mapping.get('description_column'),
                column_mapping.get('amount_column'),
                column_mapping.get('credit_column'),
                column_mapping.get('debit_column'),
                column_mapping.get('transaction_type_column'),
                column_mapping.get('account_name_column'),
                column_mapping.get('category_column')
            ] if col and col in df.columns
        ]
        
        return df[relevant_cols].head(10)
    
    def _generate_recommendations(self, column_mapping: Dict, account_preview: List[Dict], df: pd.DataFrame) -> List[str]:
        """Generate recommendations or warnings based on detection."""
        recommendations = []
        
        # Check for missing critical columns
        if not column_mapping.get('date_column'):
            recommendations.append("⚠️ Date column not detected - please select manually")
        if not column_mapping.get('amount_column') and not (column_mapping.get('credit_column') and column_mapping.get('debit_column')):
            recommendations.append("⚠️ Amount/Credit/Debit columns not detected - please select manually")
        if not column_mapping.get('description_column'):
            recommendations.append("⚠️ Description column not detected - please select manually")
        
        # Check for multiple accounts
        if len(account_preview) > 1:
            recommendations.append(f"✅ Detected {len(account_preview)} accounts - verify account types are correct")
        elif len(account_preview) == 0:
            recommendations.append("ℹ️ No account name column found - will treat all transactions as single account")
        
        # Check for account type classification
        unknown_types = [acc for acc in account_preview if acc['account_type'] == 'unknown']
        if unknown_types:
            recommendations.append(f"⚠️ {len(unknown_types)} accounts have unknown type - please classify manually")
        
        # Check data quality
        if df.isnull().sum().sum() > len(df) * 0.2:  # More than 20% nulls
            recommendations.append("⚠️ High number of missing values detected - verify data quality")
        
        return recommendations
