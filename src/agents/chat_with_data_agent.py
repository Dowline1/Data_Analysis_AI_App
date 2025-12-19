"""Conversational agent that answers user questions about their banking data."""

from __future__ import annotations

from typing import Dict, List, Any
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChatWithDataAgent:
    """LLM agent that reasons over structured metrics, transactions, and insights."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.4,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior financial advisor. Use ONLY the supplied data to answer questions. "
                    "Cite concrete numbers when possible and keep answers under 180 words."
                ),
                (
                    "human",
                    "Context JSON:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                ),
            ]
        )

    def answer_question(
        self,
        question: str,
        transactions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        category_summary: Dict[str, Any],
        subscriptions: Dict[str, Any],
        expert_report: Dict[str, Any],
        insights: Dict[str, List[str]]
    ) -> str:
        """Generate an answer grounded in the latest analysis."""
        if not question.strip():
            return "Please ask a question about your finances."

        context = {
            'metrics': metrics,
            'category_summary': category_summary,
            'subscriptions': subscriptions,
            'expert_report': expert_report,
            'insights': insights,
            'sample_transactions': [
                {
                    'date': str(tx.get('date')),
                    'description': tx.get('description'),
                    'amount': tx.get('amount'),
                    'category': tx.get('category')
                }
                for tx in transactions[:50]
            ],
        }

        chain = self.prompt | self.llm
        try:
            response = chain.invoke({
                'context': json.dumps(context, ensure_ascii=False)[:6000],
                'question': question,
            })
            return response.content.strip()
        except Exception as exc:
            logger.error(f"Chat agent failed: {exc}")
            return "Sorry, I couldn't process that question right now."