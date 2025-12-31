"""
LangGraph-based Streamlit Application

Uses LangGraph with HITL via interrupt() and Command(resume=...).
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from langgraph.types import Command
from src.graph.workflow import create_workflow, create_initial_state
from src.graph.state import AnalysisState
from src.agents.chat_with_data_agent import ChatWithDataAgent

# Sample file path
SAMPLE_FILE = ROOT_DIR / "data" / "sample_statements" / "sample_statement.xlsx"


@st.cache_resource
def get_workflow():
    """Get the compiled LangGraph workflow."""
    return create_workflow()


@st.cache_resource
def get_chat_agent() -> ChatWithDataAgent:
    """Get chat agent for Q&A."""
    return ChatWithDataAgent()


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "current_state" not in st.session_state:
        st.session_state.current_state = None
    if "interrupt_data" not in st.session_state:
        st.session_state.interrupt_data = None
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "file_bytes" not in st.session_state:
        st.session_state.file_bytes = None
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "tmp_path" not in st.session_state:
        st.session_state.tmp_path = None


def reset_workflow_state():
    """Clear workflow state for new file."""
    st.session_state.current_state = None
    st.session_state.interrupt_data = None
    st.session_state.analysis_complete = False
    st.session_state.file_bytes = None
    st.session_state.filename = None
    st.session_state.thread_id = str(uuid.uuid4())  # New thread for new file


def run_workflow(workflow, input_data: Any, config: Dict) -> Dict:
    """
    Run workflow and handle interrupts.
    
    Returns state dict with interrupt_data if interrupted.
    """
    result = {"state": None, "interrupted": False, "interrupt_data": None}
    
    try:
        print(f"DEBUG: Starting workflow stream...")
        for event in workflow.stream(input_data, config, stream_mode="values"):
            result["state"] = event
            print(f"DEBUG: Got event with keys: {event.keys() if event else None}")
        
        # Check if we hit an interrupt
        graph_state = workflow.get_state(config)
        print(f"DEBUG: Graph state next: {graph_state.next}")
        print(f"DEBUG: Graph state tasks: {graph_state.tasks}")
        
        # Check for interrupt in the state
        if graph_state.next:
            result["interrupted"] = True
            # Extract interrupt data from tasks
            for task in graph_state.tasks:
                print(f"DEBUG: Task: {task}")
                if hasattr(task, 'interrupts') and task.interrupts:
                    result["interrupt_data"] = task.interrupts[0].value
                    print(f"DEBUG: Found interrupt data: {result['interrupt_data']}")
                    break
        else:
            result["interrupted"] = False
            
    except Exception as e:
        st.error(f"Workflow error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    
    return result


def render_schema_confirmation(interrupt_data: Dict):
    """Render schema confirmation UI."""
    st.subheader("📋 Schema Detection - Human Review Required")
    st.info("👤 **Human-in-the-Loop Checkpoint**: Please verify the detected schema before proceeding.")
    
    schema_info = interrupt_data.get("schema_info", {})
    
    # Show detected columns
    st.write("**Detected Columns:**")
    columns = schema_info.get("columns", {})
    col1, col2 = st.columns(2)
    with col1:
        for col_type, col_name in columns.items():
            if col_name:
                st.success(f"✓ {col_type.title()}: {col_name}")
            else:
                st.warning(f"⚠ {col_type.title()}: Not detected")
    
    # Show warnings/recommendations
    warnings = schema_info.get("warnings", [])
    recommendations = schema_info.get("recommendations", [])
    
    with col2:
        if warnings:
            for w in warnings:
                st.warning(w)
        if recommendations:
            for r in recommendations:
                st.info(r)
    
    # Show accounts with override options
    st.write("**Detected Accounts:**")
    accounts = schema_info.get("accounts", [])
    schema_overrides = {}
    
    if accounts:
        for i, acc in enumerate(accounts):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{acc['account_name']}**")
                st.caption(f"{acc['transaction_count']} transactions")
            
            with col2:
                type_emoji = {"credit_card": "💳", "checking": "🏦", "current": "🏦", "savings": "💰"}.get(acc['account_type'], "❓")
                st.write(f"{type_emoji} {acc['account_type'].replace('_', ' ').title()}")
            
            with col3:
                new_type = st.selectbox(
                    "Override",
                    ["credit_card", "checking", "current", "savings"],
                    index=["credit_card", "checking", "current", "savings"].index(acc['account_type']) 
                          if acc['account_type'] in ["credit_card", "checking", "current", "savings"] else 0,
                    key=f"acc_type_{i}"
                )
                if new_type != acc['account_type']:
                    schema_overrides[acc['account_name']] = new_type
    else:
        st.info("No separate accounts detected - treating as single account")
    
    # Confirmation button
    st.divider()
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("✅ Confirm Schema", type="primary", use_container_width=True):
            # Resume workflow with user response
            workflow = get_workflow()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            resume_data = Command(resume={
                "confirmed": True,
                "overrides": schema_overrides
            })
            
            with st.spinner("Processing transactions..."):
                result = run_workflow(workflow, resume_data, config)
                st.session_state.current_state = result["state"]
                
                if result["interrupted"]:
                    st.session_state.interrupt_data = result["interrupt_data"]
                else:
                    st.session_state.interrupt_data = None
                    st.session_state.analysis_complete = True
            
            st.rerun()
    
    with col2:
        st.caption(f"{len(schema_overrides)} override(s)" if schema_overrides else "No overrides")


def render_subscription_confirmation(interrupt_data: Dict):
    """Render subscription confirmation UI."""
    st.subheader("🔄 Subscription Detection - Human Review Required")
    st.info("👤 **Human-in-the-Loop Checkpoint**: Please confirm detected subscriptions.")
    
    subscriptions = interrupt_data.get("subscriptions", [])
    
    if not subscriptions:
        st.info("No subscriptions detected")
        if st.button("Continue to Analysis", type="primary"):
            workflow = get_workflow()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            resume_data = Command(resume={"selections": {}, "notes": {}})
            
            with st.spinner("Generating analysis..."):
                result = run_workflow(workflow, resume_data, config)
                st.session_state.current_state = result["state"]
                st.session_state.interrupt_data = None
                st.session_state.analysis_complete = True
            
            st.rerun()
        return
    
    st.write(f"Found {len(subscriptions)} potential subscriptions:")
    
    selections = {}
    notes = {}
    
    for i, sub in enumerate(subscriptions):
        with st.expander(f"💳 {sub['merchant']} - ${sub['amount']:.2f} ({sub['frequency']})", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Category:** {sub.get('category', 'Unknown')}")
                st.write(f"**Amount:** ${sub['amount']:.2f}")
                st.write(f"**Frequency:** {sub['frequency']}")
                
                note = st.text_input("Note (optional)", key=f"sub_note_{i}", placeholder="e.g., Cancelled")
                if note:
                    notes[sub['merchant']] = note
            
            with col2:
                confirmed = st.checkbox("Confirm", value=True, key=f"sub_confirm_{i}")
                selections[sub['merchant']] = confirmed
    
    # Confirmation button
    st.divider()
    confirmed_count = sum(1 for v in selections.values() if v)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("✅ Confirm Subscriptions", type="primary", use_container_width=True):
            workflow = get_workflow()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            resume_data = Command(resume={
                "selections": selections,
                "notes": notes
            })
            
            with st.spinner("Generating analysis and insights..."):
                result = run_workflow(workflow, resume_data, config)
                st.session_state.current_state = result["state"]
                st.session_state.interrupt_data = None
                st.session_state.analysis_complete = True
            
            st.rerun()
    
    with col2:
        st.caption(f"{confirmed_count} of {len(subscriptions)} confirmed")


def _render_list_block(title: str, rows: List[str]):
    """Render a list block with title."""
    st.markdown(f"**{title}**")
    if rows:
        for item in rows:
            st.write(f"- {item}")
    else:
        st.caption("No items")


def _render_metrics_dashboard(transactions: List[Dict]):
    """Render financial metrics dashboard."""
    st.subheader("📊 Financial Snapshot")
    
    if not transactions:
        st.info("No transactions available for metrics.")
        return
    
    tx_df = pd.DataFrame(transactions)
    
    # Check if we have multiple accounts
    has_accounts = 'account_type' in tx_df.columns and tx_df['account_type'].notna().any()
    
    if has_accounts:
        st.info("💡 Each account is analyzed separately with appropriate sign conventions.")
        
        for account_type in tx_df['account_type'].unique():
            if pd.isna(account_type):
                continue
                
            account_df = tx_df[tx_df['account_type'] == account_type]
            st.markdown(f"### 📊 {str(account_type).replace('_', ' ').title()}")
            st.caption(f"{len(account_df)} transactions")
            
            total_income = account_df[account_df['amount'] > 0]['amount'].sum()
            total_expenses = abs(account_df[account_df['amount'] < 0]['amount'].sum())
            net_balance = total_income - total_expenses
            
            if 'credit' in str(account_type).lower():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Charges", f"${total_expenses:,.2f}", help="Purchases and charges")
                c2.metric("Total Payments", f"${total_income:,.2f}", help="Payments made")
                c3.metric("Net Balance", f"${-net_balance:,.2f}", 
                         help="Current debt (positive = money owed)")
                payment_ratio = (total_income / total_expenses * 100) if total_expenses > 0 else 0
                c4.metric("Payment Ratio", f"{payment_ratio:.1f}%", help="Payments as % of charges")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Deposits", f"${total_income:,.2f}", help="Money in")
                c2.metric("Total Withdrawals", f"${total_expenses:,.2f}", help="Money out")
                c3.metric("Net Change", f"${net_balance:,.2f}",
                         help="Change in account balance",
                         delta=f"${net_balance:,.2f}",
                         delta_color="normal" if net_balance > 0 else "inverse")
            
            st.divider()
    else:
        # Single account summary
        total_income = tx_df[tx_df['amount'] > 0]['amount'].sum()
        total_expenses = abs(tx_df[tx_df['amount'] < 0]['amount'].sum())
        net_balance = total_income - total_expenses
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Income", f"${total_income:,.2f}")
        c2.metric("Total Expenses", f"${total_expenses:,.2f}")
        c3.metric("Net Balance", f"${net_balance:,.2f}",
                 delta=f"${net_balance:,.2f}",
                 delta_color="normal" if net_balance > 0 else "inverse")


def _render_transactions(transactions: List[Dict]):
    """Render transactions table."""
    st.subheader("📋 Parsed Transactions")
    
    if not transactions:
        st.warning("No transactions were extracted.")
        return
    
    tx_df = pd.DataFrame(transactions)
    tx_df['date'] = pd.to_datetime(tx_df['date']).dt.date
    
    # Account summary
    if 'account_type' in tx_df.columns and tx_df['account_type'].notna().any():
        account_counts = tx_df.groupby('account_type').size()
        account_info = ', '.join([f"{str(acct).replace('_', ' ').title()} ({count})" 
                                  for acct, count in account_counts.items()])
        st.info(f"📊 **Accounts:** {account_info}")
    
    st.caption(f"Showing first {min(len(tx_df), 200)} of {len(tx_df)} rows")
    display_cols = ['date', 'description', 'amount', 'category']
    if 'account_type' in tx_df.columns:
        display_cols.append('account_type')
    st.dataframe(tx_df[display_cols].head(200), use_container_width=True)


def _render_category_overview(transactions: List[Dict]):
    """Render category breakdown with drill-down."""
    st.subheader("📁 Category Overview")
    
    if not transactions:
        st.caption("No categorized transactions yet.")
        return
    
    tx_df = pd.DataFrame(transactions)
    
    # Check for multiple accounts
    has_accounts = 'account_type' in tx_df.columns and tx_df['account_type'].notna().any()
    
    if has_accounts:
        account_types = ['All Accounts'] + sorted([str(a) for a in tx_df['account_type'].dropna().unique()])
        selected_account = st.selectbox("Filter by account", account_types, key="category_account_filter")
        
        if selected_account != 'All Accounts':
            tx_df = tx_df[tx_df['account_type'] == selected_account]
            st.caption(f"Showing categories for: **{selected_account.replace('_', ' ').title()}**")
    
    # Calculate category summary
    if 'category' in tx_df.columns:
        category_data = []
        for category in tx_df['category'].dropna().unique():
            cat_txs = tx_df[tx_df['category'] == category]
            category_data.append({
                'category': category,
                'total': cat_txs['amount'].sum(),
                'count': len(cat_txs),
                'average': cat_txs['amount'].mean()
            })
        
        summary_df = pd.DataFrame(category_data).sort_values('total')
        st.dataframe(summary_df, use_container_width=True)
        
        # Drill-down selector
        if category_data:
            selected_category = st.selectbox(
                "Drill into a category",
                options=[row['category'] for row in category_data],
                index=0
            )
            
            filtered = tx_df[tx_df['category'] == selected_category]
            if len(filtered) > 0:
                st.caption(f"Recent transactions for {selected_category}")
                cat_df = filtered.sort_values('date', ascending=False).head(20)
                cat_df['date'] = pd.to_datetime(cat_df['date']).dt.date
                st.dataframe(cat_df[['date', 'description', 'amount']], use_container_width=True)


def _render_visualizations(transactions: List[Dict]):
    """Render charts and visualizations."""
    st.subheader("📈 Visual Insights")
    
    if not transactions:
        st.caption("Upload a statement to see charts.")
        return
    
    tx_df = pd.DataFrame(transactions).copy()
    tx_df['date'] = pd.to_datetime(tx_df['date'])
    
    # Check for multiple accounts
    has_accounts = 'account_type' in tx_df.columns and tx_df['account_type'].notna().any()
    
    if has_accounts:
        account_types = ['All Accounts'] + sorted([str(a) for a in tx_df['account_type'].dropna().unique()])
        selected_account = st.selectbox("Filter by account", account_types, key="viz_account_filter")
        
        if selected_account != 'All Accounts':
            tx_df = tx_df[tx_df['account_type'] == selected_account]
    
    color_scheme = px.colors.qualitative.Set2
    
    col1, col2 = st.columns(2)
    
    # Spending by category pie chart
    expenses_df = tx_df[tx_df['amount'] < 0].copy()
    if len(expenses_df) > 0 and 'category' in expenses_df.columns:
        expenses_df['amount_abs'] = expenses_df['amount'].abs()
        spending_by_category = expenses_df.groupby('category')['amount_abs'].sum().to_dict()
        
        if spending_by_category:
            pie_fig = px.pie(
                names=list(spending_by_category.keys()),
                values=list(spending_by_category.values()),
                title="<b>Spending by Category</b>",
                color_discrete_sequence=color_scheme,
                hole=0.4
            )
            pie_fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>$%{value:.2f}<br>%{percent}<extra></extra>'
            )
            col1.plotly_chart(pie_fig, use_container_width=True)
    else:
        col1.info("No spending breakdown available.")
    
    # Cumulative balance line chart
    daily = tx_df.groupby('date')['amount'].sum().reset_index()
    daily = daily.sort_values('date')
    daily['cumulative_balance'] = daily['amount'].cumsum()
    
    line_fig = px.line(
        daily, x='date', y='cumulative_balance',
        title='<b>Cumulative Balance Over Time</b>'
    )
    line_fig.update_traces(
        line_shape='spline',
        line=dict(color='#636EFA', width=3),
        fill='tozeroy',
        fillcolor='rgba(99, 110, 250, 0.1)'
    )
    line_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Balance ($)",
        hovermode='x unified'
    )
    col2.plotly_chart(line_fig, use_container_width=True)
    
    # Monthly summary bar chart
    tx_df['month'] = tx_df['date'].dt.to_period('M').dt.to_timestamp()
    summary = tx_df.groupby('month')['amount'].sum().reset_index()
    summary = summary.sort_values('month')
    summary['cumulative'] = summary['amount'].cumsum()
    summary['color'] = summary['amount'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    
    bar_fig = px.bar(
        summary, x='month', y='amount',
        title='<b>Monthly Net Position</b>',
        color='color',
        color_discrete_map={'Positive': '#00CC96', 'Negative': '#EF553B'}
    )
    bar_fig.update_traces(showlegend=False, hovertemplate='Monthly: $%{y:.2f}<extra></extra>')
    bar_fig.add_scatter(
        x=summary['month'],
        y=summary['cumulative'],
        mode='lines+markers',
        name='Cumulative Balance',
        line=dict(color='#FFA500', width=4),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='Cumulative: $%{y:.2f}<extra></extra>'
    )
    bar_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(bar_fig, use_container_width=True)


def _render_subscriptions(subscriptions: List[Dict]):
    """Render confirmed subscriptions."""
    st.subheader("🔄 Confirmed Subscriptions")
    
    confirmed_subs = [s for s in subscriptions if s.get("confirmed")]
    
    if not confirmed_subs:
        st.info("No subscriptions were confirmed.")
        return
    
    # Calculate monthly cost
    monthly_cost = 0
    for sub in confirmed_subs:
        freq = sub.get('frequency', '').lower()
        amount = abs(sub.get('amount', 0))
        if 'month' in freq:
            monthly_cost += amount
        elif 'year' in freq or 'annual' in freq:
            monthly_cost += amount / 12
        elif 'quarter' in freq:
            monthly_cost += amount / 3
        elif 'semi' in freq:
            monthly_cost += amount / 6
        elif 'bi-week' in freq:
            monthly_cost += amount * 2
    
    m1, m2 = st.columns(2)
    m1.metric("Confirmed Subscriptions", len(confirmed_subs))
    m2.metric("Est. Monthly Cost", f"${monthly_cost:.2f}")
    
    # Display table
    sub_data = []
    for sub in confirmed_subs:
        sub_data.append({
            'Merchant': sub.get('merchant', 'Unknown'),
            'Amount': f"${abs(sub.get('amount', 0)):.2f}",
            'Frequency': sub.get('frequency', 'Unknown'),
            'Category': sub.get('category', 'Subscription')
        })
    
    sub_df = pd.DataFrame(sub_data)
    st.dataframe(sub_df, use_container_width=True)


def _render_expert_report(expert_report):
    """Render expert analysis report."""
    st.subheader("🧠 Expert AI Assessment")
    
    if not expert_report:
        st.caption("No expert analysis available.")
        return
    
    if isinstance(expert_report, str):
        st.markdown(expert_report)
        return
    
    if isinstance(expert_report, dict):
        # Overall summary
        summary = expert_report.get('overall_summary', '')
        if summary:
            st.write(summary)
        
        cols = st.columns(2)
        with cols[0]:
            _render_list_block("Subscription Findings", expert_report.get('subscription_findings', []))
            _render_list_block("Credit Card Findings", expert_report.get('credit_card_findings', []))
            _render_list_block("Recommended Actions", expert_report.get('recommended_actions', []))
        with cols[1]:
            _render_list_block("Transfer Findings", expert_report.get('transfer_findings', []))
            _render_list_block("Risk Alerts", expert_report.get('risk_alerts', []))


def _render_advisor_briefing(insights: Dict):
    """Render advisor briefing with expandable sections."""
    st.subheader("💼 Advisor Briefing")
    
    if not insights:
        st.caption("Insights will appear once analysis completes.")
        return
    
    if isinstance(insights, str):
        st.markdown(insights)
        return
    
    if isinstance(insights, dict):
        for section, messages in insights.items():
            pretty_name = section.replace('_', ' ').title()
            with st.expander(pretty_name, expanded=False):
                if messages and isinstance(messages, list):
                    for message in messages:
                        st.write(f"- {message}")
                elif messages:
                    st.write(messages)
                else:
                    st.caption("No insights for this section.")


def render_analysis_results():
    """Render comprehensive analysis results - matching original app.py quality."""
    state = st.session_state.current_state
    if not state:
        return
    
    transactions = state.get("transactions", [])
    subscriptions = state.get("detected_subscriptions", [])
    expert_report = state.get("expert_report")
    react_analysis = state.get("react_analysis")
    expert_insights = state.get("expert_insights")
    
    # Convert transactions to dict format if needed
    tx_list = [dict(tx) if hasattr(tx, 'items') else tx for tx in transactions]
    sub_list = [dict(s) if hasattr(s, 'items') else s for s in subscriptions]
    
    # Summary metrics at top
    st.success(f"✅ Analysis complete for {st.session_state.filename}")
    
    info_cols = st.columns(3)
    confirmed_subs = [s for s in sub_list if s.get("confirmed")]
    info_cols[0].metric("Transactions", len(tx_list))
    info_cols[1].metric("Confirmed Subscriptions", len(confirmed_subs))
    
    # Calculate monthly subscription cost
    monthly_cost = 0
    for sub in confirmed_subs:
        freq = sub.get('frequency', '').lower()
        amount = abs(sub.get('amount', 0))
        if 'month' in freq:
            monthly_cost += amount
        elif 'year' in freq or 'annual' in freq:
            monthly_cost += amount / 12
        elif 'quarter' in freq:
            monthly_cost += amount / 3
    info_cols[2].metric("Est. Monthly Subscription Cost", f"${monthly_cost:.2f}")
    
    # Render all sections
    _render_metrics_dashboard(tx_list)
    _render_transactions(tx_list)
    _render_category_overview(tx_list)
    _render_visualizations(tx_list)
    _render_subscriptions(sub_list)
    _render_expert_report(expert_report)
    
    # ReAct Analysis
    if react_analysis:
        st.subheader("🤖 AI Agent Analysis (ReAct)")
        st.markdown(react_analysis)
    
    # Advisor briefing
    if expert_insights:
        _render_advisor_briefing(expert_insights)


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Bank Statement Analyzer", page_icon="💰", layout="wide")
    
    initialize_session_state()
    
    st.title("💰 Bank Statement Analyzer")
    st.caption("Powered by LangGraph with Human-in-the-Loop")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Statement Input")
        uploaded_file = st.file_uploader("Upload statement", type=["csv", "xlsx", "xls"])
        use_sample = st.button("📊 Use Sample Statement", use_container_width=True)
        
        st.divider()
        
        # Debug info
        st.caption("**Debug Info:**")
        st.caption(f"Thread: {st.session_state.thread_id[:8]}...")
        st.caption(f"State: {'Yes' if st.session_state.current_state else 'No'}")
        st.caption(f"Interrupt: {'Yes' if st.session_state.interrupt_data else 'No'}")
        st.caption(f"Complete: {st.session_state.analysis_complete}")
    
    # Handle file selection
    if uploaded_file is not None:
        if st.session_state.filename != uploaded_file.name:
            reset_workflow_state()
            st.session_state.file_bytes = uploaded_file.getvalue()
            st.session_state.filename = uploaded_file.name
    elif use_sample:
        if SAMPLE_FILE.exists() and st.session_state.filename != SAMPLE_FILE.name:
            reset_workflow_state()
            st.session_state.file_bytes = SAMPLE_FILE.read_bytes()
            st.session_state.filename = SAMPLE_FILE.name
            st.rerun()
    
    # Process file
    if st.session_state.file_bytes is not None:
        # Create temp file if needed
        if st.session_state.tmp_path is None or not Path(st.session_state.tmp_path).exists():
            suffix = Path(st.session_state.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(st.session_state.file_bytes)
                st.session_state.tmp_path = tmp.name
        
        # Start workflow if not started
        if st.session_state.current_state is None and not st.session_state.interrupt_data:
            workflow = get_workflow()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            initial_state = create_initial_state(st.session_state.tmp_path)
            
            with st.spinner("Analyzing file and detecting schema..."):
                result = run_workflow(workflow, initial_state, config)
                st.session_state.current_state = result["state"]
                
                if result["interrupted"]:
                    st.session_state.interrupt_data = result["interrupt_data"]
            
            st.rerun()
        
        # Render appropriate UI based on state
        if st.session_state.interrupt_data:
            interrupt_type = st.session_state.interrupt_data.get("type", "unknown")
            
            if interrupt_type == "schema_confirmation":
                render_schema_confirmation(st.session_state.interrupt_data)
            elif interrupt_type == "subscription_confirmation":
                render_subscription_confirmation(st.session_state.interrupt_data)
            else:
                st.warning(f"Unknown interrupt type: {interrupt_type}")
                st.json(st.session_state.interrupt_data)
        
        elif st.session_state.analysis_complete:
            render_analysis_results()
            
            # Chat interface
            st.divider()
            st.subheader("💬 Chat with Your Data")
            
            user_question = st.chat_input("Ask about your finances...")
            if user_question:
                agent = get_chat_agent()
                state = st.session_state.current_state
                
                with st.chat_message("user"):
                    st.write(user_question)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Prepare category summary
                        transactions = state.get("transactions", [])
                        category_summary = {}
                        if transactions:
                            tx_df = pd.DataFrame(transactions)
                            if "category" in tx_df.columns and "amount" in tx_df.columns:
                                category_summary = tx_df.groupby("category")["amount"].agg(["sum", "count"]).to_dict()
                        
                        response = agent.answer_question(
                            question=user_question,
                            transactions=transactions,
                            metrics={"account_metrics": state.get("account_metrics", [])},
                            category_summary=category_summary,
                            subscriptions={"detected": state.get("detected_subscriptions", [])},
                            expert_report=state.get("expert_report", {}),
                            insights=state.get("expert_insights", {})
                        )
                        st.write(response)
        else:
            st.info("Processing... Please wait.")
    else:
        st.info("👆 Upload a bank statement or use the sample to begin analysis")


if __name__ == "__main__":
    main()
