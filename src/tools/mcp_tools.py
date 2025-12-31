"""
MCP-Compliant Custom Tools for Bank Statement Analysis

This module provides Model Context Protocol (MCP) compliant tools
that can be used by LLM agents in the analysis workflow.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import pandas as pd
from datetime import datetime


class FileParserTool(BaseModel):
    """
    MCP Tool: Parse bank statement file.
    
    Input Schema:
        file_path: Path to CSV/Excel file
        
    Output Schema:
        columns: List of column names
        row_count: Number of rows
        sample_data: First 5 rows
    """
    
    name: str = "file_parser"
    description: str = "Parse bank statement file and extract structure"
    
    def __call__(self, file_path: str) -> Dict[str, Any]:
        """Execute the file parser tool."""
        try:
            # Determine file type
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return {
                    "error": "Unsupported file format. Use CSV or Excel."
                }
            
            return {
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "sample_data": df.head(5).to_dict(orient='records'),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
        except Exception as e:
            return {"error": f"Failed to parse file: {str(e)}"}


class TransactionFilterTool(BaseModel):
    """
    MCP Tool: Filter transactions by criteria.
    
    Input Schema:
        transactions: List of transaction dicts
        filters: Dict of filter criteria
            - min_amount: float
            - max_amount: float
            - categories: List[str]
            - account_types: List[str]
            - date_from: str (ISO format)
            - date_to: str (ISO format)
            
    Output Schema:
        filtered_transactions: List of matching transactions
        count: Number of matches
    """
    
    name: str = "transaction_filter"
    description: str = "Filter transactions based on specified criteria"
    
    def __call__(
        self, 
        transactions: List[Dict], 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the transaction filter tool."""
        if not filters:
            return {
                "filtered_transactions": transactions,
                "count": len(transactions)
            }
        
        filtered = transactions
        
        # Apply amount filters
        if "min_amount" in filters:
            filtered = [
                tx for tx in filtered 
                if abs(tx.get("amount", 0)) >= filters["min_amount"]
            ]
        
        if "max_amount" in filters:
            filtered = [
                tx for tx in filtered 
                if abs(tx.get("amount", 0)) <= filters["max_amount"]
            ]
        
        # Apply category filter
        if "categories" in filters:
            filtered = [
                tx for tx in filtered 
                if tx.get("category") in filters["categories"]
            ]
        
        # Apply account type filter
        if "account_types" in filters:
            filtered = [
                tx for tx in filtered 
                if tx.get("account_type") in filters["account_types"]
            ]
        
        # Apply date filters
        if "date_from" in filters:
            date_from = datetime.fromisoformat(filters["date_from"])
            filtered = [
                tx for tx in filtered 
                if datetime.fromisoformat(tx.get("date", "")) >= date_from
            ]
        
        if "date_to" in filters:
            date_to = datetime.fromisoformat(filters["date_to"])
            filtered = [
                tx for tx in filtered 
                if datetime.fromisoformat(tx.get("date", "")) <= date_to
            ]
        
        return {
            "filtered_transactions": filtered,
            "count": len(filtered),
            "original_count": len(transactions)
        }


class AggregationTool(BaseModel):
    """
    MCP Tool: Aggregate transaction data.
    
    Input Schema:
        transactions: List of transaction dicts
        group_by: Field to group by (category, account_type, date)
        aggregation: Type of aggregation (sum, count, avg)
        
    Output Schema:
        aggregated_data: Dict of grouped results
        total: Overall total
    """
    
    name: str = "transaction_aggregator"
    description: str = "Aggregate transaction data by grouping criteria"
    
    def __call__(
        self,
        transactions: List[Dict],
        group_by: str,
        aggregation: str = "sum"
    ) -> Dict[str, Any]:
        """Execute the aggregation tool."""
        if not transactions:
            return {"aggregated_data": {}, "total": 0}
        
        # Group transactions
        grouped = {}
        for tx in transactions:
            key = tx.get(group_by, "Unknown")
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(tx)
        
        # Apply aggregation
        results = {}
        total = 0
        
        for key, txs in grouped.items():
            if aggregation == "sum":
                value = sum(abs(tx.get("amount", 0)) for tx in txs)
            elif aggregation == "count":
                value = len(txs)
            elif aggregation == "avg":
                value = sum(abs(tx.get("amount", 0)) for tx in txs) / len(txs)
            else:
                value = len(txs)
            
            results[key] = value
            total += value if aggregation != "count" else 1
        
        return {
            "aggregated_data": results,
            "total": total,
            "group_count": len(results)
        }


class TrendAnalysisTool(BaseModel):
    """
    MCP Tool: Analyze spending/income trends over time.
    
    Input Schema:
        transactions: List of transaction dicts
        period: Grouping period (daily, weekly, monthly)
        metric: What to analyze (spending, income, net)
        
    Output Schema:
        trend_data: Time series data
        trend_direction: increasing/decreasing/stable
        growth_rate: Percentage change
    """
    
    name: str = "trend_analyzer"
    description: str = "Analyze financial trends over time periods"
    
    def __call__(
        self,
        transactions: List[Dict],
        period: str = "monthly",
        metric: str = "spending"
    ) -> Dict[str, Any]:
        """Execute the trend analysis tool."""
        if not transactions:
            return {
                "trend_data": {},
                "trend_direction": "stable",
                "growth_rate": 0
            }
        
        # Convert to DataFrame for easier time series handling
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = pd.to_numeric(df['amount'])
        
        # Set time period
        if period == "daily":
            df['period'] = df['date'].dt.date
        elif period == "weekly":
            df['period'] = df['date'].dt.to_period('W')
        else:  # monthly
            df['period'] = df['date'].dt.to_period('M')
        
        # Calculate metric by period
        if metric == "spending":
            grouped = df[df['amount'] < 0].groupby('period')['amount'].sum().abs()
        elif metric == "income":
            grouped = df[df['amount'] > 0].groupby('period')['amount'].sum()
        else:  # net
            grouped = df.groupby('period')['amount'].sum()
        
        trend_data = grouped.to_dict()
        
        # Calculate trend direction
        if len(grouped) < 2:
            trend_direction = "stable"
            growth_rate = 0
        else:
            first_half = grouped.iloc[:len(grouped)//2].mean()
            second_half = grouped.iloc[len(grouped)//2:].mean()
            
            if second_half > first_half * 1.1:
                trend_direction = "increasing"
            elif second_half < first_half * 0.9:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            growth_rate = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
        
        return {
            "trend_data": {str(k): float(v) for k, v in trend_data.items()},
            "trend_direction": trend_direction,
            "growth_rate": float(growth_rate)
        }


# Tool registry for MCP
MCP_TOOLS = {
    "file_parser": FileParserTool(),
    "transaction_filter": TransactionFilterTool(),
    "transaction_aggregator": AggregationTool(),
    "trend_analyzer": TrendAnalysisTool()
}


def get_tool(tool_name: str):
    """Get an MCP tool by name."""
    return MCP_TOOLS.get(tool_name)


def list_available_tools() -> List[str]:
    """List all available MCP tools."""
    return list(MCP_TOOLS.keys())
