"""Expert banking insights agent that lets the LLM review the full ledger."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.state.app_state import Transaction
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ExpertAssessment(BaseModel):
    """Structured response returned by the LLM expert."""

    overall_summary: str = Field(description="Narrative overview of the customer's finances")
    subscription_findings: List[str] = Field(
        default_factory=list,
        description="Observations about recurring services or memberships",
    )
    transfer_findings: List[str] = Field(
        default_factory=list,
        description="Insights about internal transfers or money movement between accounts",
    )
    credit_card_findings: List[str] = Field(
        default_factory=list,
        description="How credit cards and debt pay-downs are being handled",
    )
    risk_alerts: List[str] = Field(
        default_factory=list,
        description="Potential risks, anomalies, or suspicious behaviour",
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Concrete, expert-level recommendations",
    )


class ExpertInsightsAgent:
    """Runs an LLM-only review across the full ledger when a deeper read is needed."""

    def __init__(self, max_ledger_rows: int = 400):
        self.max_ledger_rows = max_ledger_rows
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.2,
        )
        self.parser = JsonOutputParser(pydantic_object=ExpertAssessment)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior banking expert. Review the ledger and describe subscriptions, "
                    "transfers, credit-card behaviour, and risks. "
                    "Respond ONLY in valid JSON that follows these instructions:\n{format_instructions}",
                ),
                (
                    "human",
                    "Summary stats: {summary_stats}\n"
                    "Account stats: {account_stats}\n"
                    "Top merchants: {top_merchants}\n"
                    "Ledger sample (most recent first, {row_count} rows max): {ledger_sample}\n"
                    "Provide expert findings.",
                ),
            ]
        )
        logger.info("Expert insights agent ready for full-ledger analysis")

    def analyze_transactions(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Send the entire data set to the LLM for expert review."""
        if not transactions:
            logger.warning("ExpertInsightsAgent received no transactions")
            return {
                "overall_summary": "No data provided",
                "subscription_findings": [],
                "transfer_findings": [],
                "credit_card_findings": [],
                "risk_alerts": [],
                "recommended_actions": [],
            }

        context = self._build_context(transactions)
        chain = self.prompt | self.llm | self.parser
        try:
            assessment = chain.invoke(
                {
                    "format_instructions": self.parser.get_format_instructions(),
                    "summary_stats": json.dumps(context["summary_stats"], ensure_ascii=True),
                    "account_stats": json.dumps(context["account_stats"], ensure_ascii=True),
                    "top_merchants": json.dumps(context["top_merchants"], ensure_ascii=True),
                    "ledger_sample": json.dumps(context["ledger_sample"], ensure_ascii=True),
                    "row_count": len(context["ledger_sample"]),
                }
            )
            if isinstance(assessment, dict):
                assessment = ExpertAssessment(**assessment)
            return assessment.model_dump()
        except Exception as exc:
            logger.error(f"Expert insights generation failed: {exc}")
            return {
                "overall_summary": "Expert review failed",
                "subscription_findings": [],
                "transfer_findings": [],
                "credit_card_findings": [],
                "risk_alerts": [f"LLM error: {exc}"],
                "recommended_actions": [],
            }

    def _build_context(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Compress the ledger so the LLM can reason about it."""
        sorted_tx = sorted(transactions, key=lambda t: t["date"], reverse=True)
        ledger_sample = []
        for tx in sorted_tx[: self.max_ledger_rows]:
            ledger_sample.append(
                {
                    "date": self._safe_date(tx.get("date")),
                    "description": tx.get("description", ""),
                    "amount": round(float(tx.get("amount", 0.0)), 2),
                    "category": tx.get("category") or "Uncategorized",
                    "account": tx.get("account", "primary"),
                    "type": tx.get("type", "unknown"),
                    "balance": tx.get("balance"),
                }
            )

        summary_stats = self._summarize(transactions)
        account_stats = self._account_breakdown(transactions)
        top_merchants = self._top_merchants(transactions)

        return {
            "summary_stats": summary_stats,
            "account_stats": account_stats,
            "top_merchants": top_merchants,
            "ledger_sample": ledger_sample,
        }

    def _summarize(self, transactions: List[Transaction]) -> Dict[str, Any]:
        total_income = 0.0
        total_expense = 0.0
        for tx in transactions:
            amount = float(tx.get("amount", 0.0))
            if amount >= 0:
                total_income += amount
            else:
                total_expense += amount

        dates = [tx.get("date") for tx in transactions if isinstance(tx.get("date"), datetime)]
        span = {}
        if dates:
            span = {
                "start_date": min(dates).strftime("%Y-%m-%d"),
                "end_date": max(dates).strftime("%Y-%m-%d"),
                "days": (max(dates) - min(dates)).days or 1,
            }

        return {
            "transaction_count": len(transactions),
            "total_income": round(total_income, 2),
            "total_expense": round(total_expense, 2),
            "net_flow": round(total_income + total_expense, 2),
            "date_span": span,
        }

    def _account_breakdown(self, transactions: List[Transaction]) -> Dict[str, Any]:
        accounts: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "transactions": 0,
            "credits": 0.0,
            "debits": 0.0,
            "net_flow": 0.0,
        })

        for tx in transactions:
            account = tx.get("account") or tx.get("source_account") or "primary"
            amount = float(tx.get("amount", 0.0))
            accounts[account]["transactions"] += 1
            if amount >= 0:
                accounts[account]["credits"] += amount
            else:
                accounts[account]["debits"] += amount
            accounts[account]["net_flow"] += amount

        # Round for readability
        for stats in accounts.values():
            stats["credits"] = round(stats["credits"], 2)
            stats["debits"] = round(stats["debits"], 2)
            stats["net_flow"] = round(stats["net_flow"], 2)

        return accounts

    def _top_merchants(self, transactions: List[Transaction], top_n: int = 10) -> List[Dict[str, Any]]:
        counter = Counter()
        for tx in transactions:
            desc = (tx.get("description") or "").strip().lower()
            if not desc:
                continue
            counter[desc] += 1

        top_entries = []
        for description, count in counter.most_common(top_n):
            sample_amounts = [
                float(tx.get("amount", 0.0))
                for tx in transactions
                if (tx.get("description") or "").strip().lower() == description
            ]
            avg_amount = sum(sample_amounts) / len(sample_amounts) if sample_amounts else 0.0
            top_entries.append(
                {
                    "description": description,
                    "occurrences": count,
                    "average_amount": round(avg_amount, 2),
                }
            )

        return top_entries

    @staticmethod
    def _safe_date(value: Any) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        return str(value)
