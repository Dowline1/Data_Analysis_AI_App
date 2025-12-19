"""Streamlit front-end for human-in-the-loop subscription validation."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from src.agents.chat_with_data_agent import ChatWithDataAgent
from src.agents.expert_insights_agent import ExpertInsightsAgent
from src.agents.insights_generator import InsightsGeneratorAgent
from src.agents.metrics_calculator import MetricsCalculatorAgent
from src.agents.schema_mapper import SchemaMapperAgent
from src.agents.subscription_detector import SubscriptionDetectorAgent
from src.agents.transaction_categorizer import TransactionCategorizerAgent
from src.tools.file_parser import FileParser

SAMPLE_FILE = Path("data/sample_statements/sample_statement.xlsx")


@st.cache_resource(show_spinner=False)
def _get_chat_agent() -> ChatWithDataAgent:
    return ChatWithDataAgent()


@st.cache_data(show_spinner=False)
def _run_pipeline(file_bytes: bytes, filename: str) -> Dict:
    """Execute the ingestion + analysis stack for a given file payload."""
    suffix = Path(filename).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        parsed = FileParser.parse_file(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    mapper = SchemaMapperAgent()
    transactions = mapper.extract_from_dataframe(parsed['dataframe'], parsed['file_type'])

    categorizer = TransactionCategorizerAgent()
    categorized_transactions = categorizer.categorize_transactions(transactions)
    category_summary = categorizer.get_category_summary(categorized_transactions) if categorized_transactions else {}

    metrics_agent = MetricsCalculatorAgent()
    metrics = metrics_agent.calculate_all_metrics(categorized_transactions)

    detector = SubscriptionDetectorAgent(min_occurrences=2, max_day_variance=5)
    subscription_result = detector.detect_subscriptions(categorized_transactions)

    insights_agent = InsightsGeneratorAgent()
    insights = insights_agent.generate_insights(metrics, category_summary, subscription_result)

    expert_agent = ExpertInsightsAgent(max_ledger_rows=400)
    expert_report = expert_agent.analyze_transactions(categorized_transactions)

    return {
        'transactions': categorized_transactions,
        'metrics': metrics,
        'category_summary': category_summary,
        'subscription_result': subscription_result,
        'expert_report': expert_report,
        'insights': insights,
        'file_type': parsed['file_type'],
        'columns': parsed['columns'],
    }


def _render_transactions(transactions: List[Dict]):
    st.subheader("Parsed transactions")
    if not transactions:
        st.warning("No transactions were extracted from this statement.")
        return

    tx_df = pd.DataFrame(transactions)
    tx_df['date'] = pd.to_datetime(tx_df['date']).dt.date
    st.caption(f"Showing first {min(len(tx_df), 200)} of {len(tx_df)} rows")
    st.dataframe(tx_df.head(200), use_container_width=True)


def _render_subscriptions(subscription_result: Dict):
    st.subheader("AI-detected subscriptions")
    subs = subscription_result.get('subscriptions', [])
    count = subscription_result.get('count', 0)
    monthly_cost = subscription_result.get('total_subscription_cost', 0.0)

    m1, m2 = st.columns(2)
    m1.metric("Detected subscriptions", count)
    m2.metric("Monthly subscription cost", f"€{monthly_cost:.2f}")

    if not subs:
        st.info("No recurring patterns detected. Upload another statement or adjust the detection settings.")
        return subs

    table_cols = ['description', 'frequency', 'estimated_monthly_cost', 'occurrences', 'first_seen', 'last_seen', 'validated_by_llm']
    sub_df = pd.DataFrame(subs)[table_cols]
    sub_df.rename(columns={'estimated_monthly_cost': 'monthly_cost', 'validated_by_llm': 'llm_override'}, inplace=True)
    sub_df['monthly_cost'] = sub_df['monthly_cost'].map(lambda v: f"€{v:.2f}")
    st.dataframe(sub_df, use_container_width=True)

    return subs


def _render_human_review(subs: List[Dict]):
    st.subheader("Human confirmation")
    if not subs:
        st.caption("Nothing to review – the AI did not find subscriptions.")
        return

    option_map = {}
    for idx, sub in enumerate(subs):
        label = f"{sub['description']} — {sub['frequency']} (€/mo {sub['estimated_monthly_cost']:.2f})"
        option_map[label] = idx

    stored = st.session_state.get('confirmed_options', list(option_map))
    review_notes = st.session_state.get('review_notes', "")

    with st.form("hil_review"):
        confirmed = st.multiselect(
            "Select the subscriptions that look correct",
            list(option_map.keys()),
            default=stored,
        )
        notes = st.text_area("Reviewer notes", value=review_notes, placeholder="e.g. Netflix is actually bundled with Sky...")
        submitted = st.form_submit_button("Save review")

    if submitted:
        st.session_state['confirmed_options'] = confirmed
        st.session_state['review_notes'] = notes
        rejected = [opt for opt in option_map if opt not in confirmed]
        st.session_state['review_result'] = {
            'confirmed': confirmed,
            'rejected': rejected,
            'notes': notes,
        }
        st.success("Review captured – thanks!")

    if 'review_result' in st.session_state:
        result = st.session_state['review_result']
        c1, c2 = st.columns(2)
        with c1:
            st.write("✅ Confirmed")
            if result['confirmed']:
                for item in result['confirmed']:
                    st.write(f"- {item}")
            else:
                st.write("(none)")
        with c2:
            st.write("🚫 Rejected")
            if result['rejected']:
                for item in result['rejected']:
                    st.write(f"- {item}")
            else:
                st.write("(none)")
        if result['notes']:
            st.info(f"Reviewer notes: {result['notes']}")


def _render_expert_report(report: Dict):
    st.subheader("Expert AI assessment")
    st.write(report.get('overall_summary', 'No summary returned.'))

    cols = st.columns(2)
    with cols[0]:
        _render_list_block("Subscription findings", report.get('subscription_findings', []))
        _render_list_block("Credit-card findings", report.get('credit_card_findings', []))
        _render_list_block("Recommended actions", report.get('recommended_actions', []))
    with cols[1]:
        _render_list_block("Transfer findings", report.get('transfer_findings', []))
        _render_list_block("Risk alerts", report.get('risk_alerts', []))


def _render_list_block(title: str, rows: List[str]):
    st.markdown(f"**{title}**")
    if rows:
        for item in rows:
            st.write(f"- {item}")
    else:
        st.caption("No items")


def _render_metrics_dashboard(metrics: Dict):
    st.subheader("Financial snapshot")
    if not metrics:
        st.info("Metrics will appear once a statement is analyzed.")
        return

    basic = metrics.get('basic', {})
    income = metrics.get('income', {})
    spending = metrics.get('spending', {})
    health = metrics.get('health', {})

    cols = st.columns(4)
    cols[0].metric("Total income", f"€{basic.get('total_income', 0):.2f}")
    cols[1].metric("Total expenses", f"€{basic.get('total_expenses', 0):.2f}")
    cols[2].metric("Net balance", f"€{basic.get('net_balance', 0):.2f}")
    cols[3].metric("Health score", f"{health.get('health_score', 0)}/100", help=f"Rating: {health.get('rating', 'N/A')}")

    cols2 = st.columns(3)
    cols2[0].metric("Average expense", f"€{abs(spending.get('average_spend', 0)):.2f}")
    cols2[1].metric("Average income", f"€{income.get('average_income', 0):.2f}")
    cols2[2].metric("Savings rate", f"{health.get('savings_percentage', 0):.2f}%")


def _render_category_overview(category_summary: Dict, transactions: List[Dict]):
    st.subheader("Category overview")
    if not category_summary:
        st.caption("No categorized transactions yet.")
        return

    data = [
        {
            'category': category,
            'total': stats['total'],
            'count': stats['count'],
            'average': stats['average']
        }
        for category, stats in category_summary.items()
    ]
    summary_df = pd.DataFrame(data).sort_values('total')
    st.dataframe(summary_df, use_container_width=True)

    selected_category = st.selectbox(
        "Drill into a category",
        options=[row['category'] for row in data],
        index=0,
    )

    filtered = [tx for tx in transactions if tx.get('category') == selected_category]
    if filtered:
        st.caption(f"Most recent transactions for {selected_category}")
        cat_df = pd.DataFrame(filtered).sort_values('date', ascending=False).head(20)
        cat_df['date'] = pd.to_datetime(cat_df['date']).dt.date
        st.dataframe(cat_df[['date', 'description', 'amount']], use_container_width=True)


def _render_visualizations(transactions: List[Dict], metrics: Dict):
    st.subheader("Visual insights")
    if not transactions:
        st.caption("Upload a statement to see charts.")
        return

    tx_df = pd.DataFrame(transactions).copy()
    tx_df['date'] = pd.to_datetime(tx_df['date'])

    spending_by_category = metrics.get('spending', {}).get('spending_by_category', {})
    col1, col2 = st.columns(2)

    if spending_by_category:
        pie_fig = px.pie(
            names=list(spending_by_category.keys()),
            values=list(spending_by_category.values()),
            title="Spending allocation by category",
        )
        col1.plotly_chart(pie_fig, use_container_width=True)
    else:
        col1.info("No spending breakdown available.")

    daily = tx_df.groupby('date')['amount'].sum().reset_index()
    line_fig = px.line(daily, x='date', y='amount', title='Net cash flow by day')
    line_fig.update_traces(line_shape='spline')
    col2.plotly_chart(line_fig, use_container_width=True)

    tx_df['month'] = tx_df['date'].dt.to_period('M').dt.to_timestamp()
    summary = tx_df.groupby('month')['amount'].sum().reset_index()
    bar_fig = px.bar(summary, x='month', y='amount', title='Monthly net position')
    st.plotly_chart(bar_fig, use_container_width=True)


def _render_advisor_briefing(insights: Dict, expert_report: Dict):
    st.subheader("Advisor briefing")
    if expert_report:
        st.info(expert_report.get('overall_summary', ''))

    if not insights:
        st.caption("Insights will appear once analysis completes.")
        return

    for section, messages in insights.items():
        pretty_name = section.replace('_', ' ').title()
        with st.expander(pretty_name, expanded=False):
            if messages:
                for message in messages:
                    st.write(f"- {message}")
            else:
                st.caption("No insights for this section.")


def _render_chat_with_data(
    chat_agent: ChatWithDataAgent,
    transactions: List[Dict],
    metrics: Dict,
    category_summary: Dict,
    subscriptions: Dict,
    expert_report: Dict,
    insights: Dict,
):
    st.subheader("Chat with your data")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    for role, message in st.session_state['chat_history']:
        st.chat_message(role).write(message)

    prompt = st.chat_input("Ask about your finances or this statement")
    if prompt:
        st.session_state['chat_history'].append(('user', prompt))
        st.chat_message('user').write(prompt)
        with st.chat_message('assistant'):
            with st.spinner("Thinking through your data..."):
                answer = chat_agent.answer_question(
                    prompt,
                    transactions,
                    metrics,
                    category_summary,
                    subscriptions,
                    expert_report,
                    insights,
                )
            st.write(answer)
        st.session_state['chat_history'].append(('assistant', answer))


def main():
    st.set_page_config(page_title="AI Banking Review", layout="wide")
    st.title("AI-powered banking insights")
    st.caption("Upload a CSV/XLSX statement, let the agents analyze it, and confirm the detected subscriptions.")

    with st.sidebar:
        st.header("Statement input")
        uploaded_file = st.file_uploader("Upload statement", type=["csv", "xlsx"])
        use_sample = st.button("Use sample statement")
        st.caption("Tip: set GOOGLE_API_KEY in your environment before running the app.")

    pipeline_result = None
    filename = None

    if uploaded_file is not None:
        filename = uploaded_file.name
        with st.spinner("Analyzing uploaded statement..."):
            pipeline_result = _run_pipeline(uploaded_file.getvalue(), uploaded_file.name)
    elif use_sample:
        if not SAMPLE_FILE.exists():
            st.error(f"Sample file not found at {SAMPLE_FILE}")
        else:
            filename = SAMPLE_FILE.name
            with st.spinner("Analyzing sample statement..."):
                pipeline_result = _run_pipeline(SAMPLE_FILE.read_bytes(), SAMPLE_FILE.name)

    if pipeline_result is None:
        st.info("Upload a bank statement to get started.")
        return

    st.success(f"Analysis complete for {filename}")

    transactions = pipeline_result['transactions']
    metrics = pipeline_result.get('metrics', {})
    category_summary = pipeline_result.get('category_summary', {})
    subscription_result = pipeline_result['subscription_result']
    expert_report = pipeline_result['expert_report']
    insights = pipeline_result.get('insights', {})

    info_cols = st.columns(3)
    info_cols[0].metric("Transactions", len(transactions))
    info_cols[1].metric("Detected subscriptions", subscription_result.get('count', 0))
    info_cols[2].metric("Monthly subscription cost", f"€{subscription_result.get('total_subscription_cost', 0.0):.2f}")

    _render_metrics_dashboard(metrics)
    _render_transactions(transactions)
    _render_category_overview(category_summary, transactions)
    _render_visualizations(transactions, metrics)
    subs = _render_subscriptions(subscription_result)
    _render_human_review(subs)
    _render_expert_report(expert_report)
    _render_advisor_briefing(insights, expert_report)

    chat_agent = _get_chat_agent()
    _render_chat_with_data(
        chat_agent,
        transactions,
        metrics,
        category_summary,
        subscription_result,
        expert_report,
        insights,
    )


if __name__ == "__main__":
    main()
