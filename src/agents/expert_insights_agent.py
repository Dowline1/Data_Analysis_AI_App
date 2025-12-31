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

from src.graph.state import Transaction
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
                    "\n\nIMPORTANT CONTEXT FOR CREDIT CARDS:"
                    "\n- Credit card accounts use DEBT tracking (positive = debt owed, negative = credit/overpayment)"
                    "\n- A negative balance on a credit card is EXCELLENT (means overpayment/credit)"
                    "\n- Payments to credit cards (negative amounts on CC) reduce debt and are POSITIVE behavior"
                    "\n- High payment ratios (paying more than charged) indicate GOOD financial management"
                    "\n- Checking/Savings use normal tracking (positive = money in, negative = money out)"
                    "\n\nWHEN REVIEWING CREDIT CARDS:"
                    "\n1. FIRST look at the credit_card_summary to see overall account health"
                    "\n2. If net_balance is NEGATIVE or payment_ratio > 1.0, this is EXCELLENT management"
                    "\n3. Frequent payments indicate RESPONSIBLE behavior, not financial problems"
                    "\n4. Only flag as concerning if net_balance is HIGH AND POSITIVE (accumulating debt)"
                    "\n\nRespond ONLY in valid JSON that follows these instructions:\n{format_instructions}",
                ),
                (
                    "human",
                    "Summary stats: {summary_stats}\n"
                    "Account breakdown: {account_stats}\n"
                    "Credit card summary: {credit_card_summary}\n"
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
                    "credit_card_summary": json.dumps(context["credit_card_summary"], ensure_ascii=True),
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
                    "account_type": tx.get("account_type") or "unknown",
                    "account_name": tx.get("account_name") or tx.get("account") or "primary",
                    "type": tx.get("type", "unknown"),
                    "balance": tx.get("balance"),
                }
            )

        summary_stats = self._summarize(transactions)
        account_stats = self._account_breakdown(transactions)
        credit_card_summary = self._credit_card_analysis(transactions)
        top_merchants = self._top_merchants(transactions)

        return {
            "summary_stats": summary_stats,
            "account_stats": account_stats,
            "credit_card_summary": credit_card_summary,
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

    def _credit_card_analysis(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """
        Analyze credit card behavior to provide context on debt management.
        Emphasizes overall account health to help LLM understand overpayments are positive.
        
        Returns:
            Dictionary with credit card health metrics and overall assessment
        """
        # Filter for credit card transactions - check for various credit card account type patterns
        def is_credit_card_account(account_type: str) -> bool:
            if not account_type:
                return False
            account_lower = account_type.lower()
            # Match credit_card, or any type containing "card" (platinum_card, silver_card, gold_card, etc.)
            # but exclude debit_card
            return ("card" in account_lower and "debit" not in account_lower) or account_lower == "credit_card"
        
        cc_transactions = [tx for tx in transactions if is_credit_card_account(tx.get("account_type", ""))]
        
        logger.info(f"Filtered {len(cc_transactions)} credit card transactions from {len(transactions)} total")
        
        if not cc_transactions:
            logger.warning("No credit card transactions found - returning has_credit_cards: False")
            return {"has_credit_cards": False}
        
        # Calculate by credit card account
        cc_accounts = defaultdict(lambda: {
            "total_charges": 0.0,
            "total_payments": 0.0,
            "net_balance": 0.0,
            "transaction_count": 0
        })
        
        for tx in cc_transactions:
            account_name = tx.get("account_name", "Unknown Card")
            amount = float(tx.get("amount", 0.0))
            
            cc_accounts[account_name]["transaction_count"] += 1
            
            # Credit card convention: negative = charges (money spent), positive = payments made
            if amount < 0:
                cc_accounts[account_name]["total_charges"] += abs(amount)
            else:
                cc_accounts[account_name]["total_payments"] += amount
            
            # Net balance: positive = overpayment (credit), negative = debt owed
            cc_accounts[account_name]["net_balance"] += amount
        
        # Analyze each account and calculate overall health
        analysis = {
            "has_credit_cards": True,
            "overall_assessment": "",
            "accounts": {}
        }
        
        total_net_balance = 0.0
        total_payment_ratio = 0.0
        excellent_count = 0
        good_count = 0
        
        for account_name, stats in cc_accounts.items():
            payment_ratio = 0
            if stats["total_charges"] > 0:
                payment_ratio = stats["total_payments"] / stats["total_charges"]
            elif stats["total_payments"] > 0:
                payment_ratio = 2.0  # Only payments, no charges = excellent
            
            # Net balance interpretation: POSITIVE = overpayment/credit (excellent), NEGATIVE = debt owed
            health_status = "EXCELLENT"
            if stats["net_balance"] > 0:
                health_status = "EXCELLENT - Overpaid (credit balance of €{:.2f})".format(stats["net_balance"])
                excellent_count += 1
            elif stats["net_balance"] == 0:
                health_status = "EXCELLENT - Fully paid off"
                excellent_count += 1
            elif payment_ratio >= 1.0:
                health_status = "GOOD - Paying off charges (payment ratio: {:.1f}x)".format(payment_ratio)
                good_count += 1
            elif payment_ratio >= 0.8:
                health_status = "FAIR - Most charges covered"
            elif payment_ratio >= 0.5:
                health_status = "FAIR - Partial payment"
            else:
                health_status = "CONCERN - Debt accumulating"
            
            total_net_balance += stats["net_balance"]
            total_payment_ratio += payment_ratio
            
            analysis["accounts"][account_name] = {
                "total_charges": round(stats["total_charges"], 2),
                "total_payments": round(stats["total_payments"], 2),
                "net_balance": round(stats["net_balance"], 2),
                "payment_ratio": round(payment_ratio, 2),
                "health_status": health_status,
                "transaction_count": stats["transaction_count"]
            }
        
        # Generate overall assessment
        avg_payment_ratio = total_payment_ratio / len(cc_accounts)
        
        # POSITIVE net balance = customer has credit (overpaid), NEGATIVE = owes money
        if total_net_balance > 0:
            analysis["overall_assessment"] = (
                f"EXCELLENT CREDIT CARD MANAGEMENT: All accounts show credit balances (total overpayment: €{total_net_balance:.2f}). "
                f"Customer is paying MORE than they charge, maintaining positive credit on cards. This is exemplary financial behavior."
            )
        elif total_net_balance == 0:
            analysis["overall_assessment"] = (
                f"EXCELLENT CREDIT CARD MANAGEMENT: All accounts are fully paid off with zero balance. "
                f"Customer is maintaining perfect payment discipline."
            )
        elif excellent_count == len(cc_accounts):
            analysis["overall_assessment"] = (
                f"EXCELLENT CREDIT CARD MANAGEMENT: All {len(cc_accounts)} accounts show excellent payment behavior with "
                f"payment ratios averaging {avg_payment_ratio:.1f}x. Customer is responsibly managing credit card debt."
            )
        elif excellent_count + good_count == len(cc_accounts):
            analysis["overall_assessment"] = (
                f"GOOD CREDIT CARD MANAGEMENT: {excellent_count} excellent and {good_count} good accounts. "
                f"Net balance: €{total_net_balance:.2f}. Customer is managing credit responsibly."
            )
        else:
            analysis["overall_assessment"] = (
                f"MIXED CREDIT CARD MANAGEMENT: Net balance: €{total_net_balance:.2f}. "
                f"Some accounts need attention. Review individual account details."
            )
        
        # Add summary metrics
        analysis["summary"] = {
            "total_accounts": len(cc_accounts),
            "combined_net_balance": round(total_net_balance, 2),
            "average_payment_ratio": round(avg_payment_ratio, 2),
            "accounts_in_credit": sum(1 for stats in cc_accounts.values() if stats["net_balance"] > 0),
            "accounts_fully_paid": sum(1 for stats in cc_accounts.values() if stats["net_balance"] == 0),
            "accounts_with_debt": sum(1 for stats in cc_accounts.values() if stats["net_balance"] < 0),
            "excellent_accounts": excellent_count,
            "good_accounts": good_count
        }
        
        return analysis

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
