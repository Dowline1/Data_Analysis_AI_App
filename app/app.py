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
from src.agents.schema_preview_agent import SchemaPreviewAgent
from src.agents.subscription_detector import SubscriptionDetectorAgent
from src.agents.transaction_categorizer import TransactionCategorizerAgent
from src.tools.file_parser import FileParser

SAMPLE_FILE = Path("data/sample_statements/sample_statement.xlsx")


@st.cache_resource(show_spinner=False)
def _get_chat_agent() -> ChatWithDataAgent:
    return ChatWithDataAgent()


@st.cache_data(show_spinner=False)
def _get_schema_preview(file_bytes: bytes, filename: str) -> Dict:
    """Stage 1: Quick schema detection and preview for HITL confirmation."""
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

    preview_agent = SchemaPreviewAgent()
    preview = preview_agent.generate_preview(parsed['dataframe'], parsed['file_type'], filename)
    preview['parsed_data'] = parsed  # Store for later use
    
    return preview


@st.cache_data(show_spinner=False)
def _run_initial_categorization(file_bytes: bytes, filename: str) -> Dict:
    """Stage 2: Categorization and subscription detection for HITL confirmation."""
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
    transactions = mapper.extract_from_dataframe(parsed['dataframe'], parsed['file_type'], filename)

    categorizer = TransactionCategorizerAgent()
    categorized_transactions = categorizer.categorize_transactions(transactions)
    category_summary = categorizer.get_category_summary(categorized_transactions) if categorized_transactions else {}

    detector = SubscriptionDetectorAgent(min_occurrences=2, max_day_variance=5)
    subscription_result = detector.detect_subscriptions(categorized_transactions)

    return {
        'transactions': categorized_transactions,
        'category_summary': category_summary,
        'subscription_result': subscription_result,
        'file_type': parsed['file_type'],
        'columns': parsed['columns'],
    }


@st.cache_data(show_spinner=False)
def _run_full_analysis(categorized_transactions: List[Dict], confirmed_subscriptions: List[str]) -> Dict:
    """Stage 3: Full analysis with confirmed subscription data."""
    # Update subscription result with confirmed data
    filtered_subs = [
        sub for sub in st.session_state.initial_result['subscription_result']['subscriptions']
        if sub['description'] in confirmed_subscriptions
    ]
    
    subscription_result = {
        'subscriptions': filtered_subs,
        'count': len(filtered_subs),
        'total_subscription_cost': sum(s['estimated_monthly_cost'] for s in filtered_subs)
    }
    
    # Calculate metrics
    metrics_agent = MetricsCalculatorAgent()
    metrics = metrics_agent.calculate_all_metrics(categorized_transactions)

    # Generate insights with confirmed subscriptions
    insights_agent = InsightsGeneratorAgent()
    category_summary = st.session_state.initial_result['category_summary']
    insights = insights_agent.generate_insights(metrics, category_summary, subscription_result)

    # Expert analysis with confirmed subscriptions
    expert_agent = ExpertInsightsAgent(max_ledger_rows=400)
    expert_report = expert_agent.analyze_transactions(categorized_transactions)

    return {
        'transactions': categorized_transactions,
        'metrics': metrics,
        'category_summary': category_summary,
        'subscription_result': subscription_result,
        'expert_report': expert_report,
        'insights': insights,
    }


def _render_schema_confirmation(preview: Dict, filename: str):
    """
    Stage 1 UI: Show detected schema and accounts, allow human confirmation before full analysis.
    This implements the HITL element for schema/account validation.
    """
    st.success(f"Schema detected for: **{filename}**")
    st.markdown("### 🔍 Schema & Account Detection")
    st.info("👤 **Human-In-The-Loop Check**: Please verify the detected columns and account types are correct before proceeding with full analysis.")
    
    # Show recommendations first
    recommendations = preview.get('recommendations', [])
    if recommendations:
        st.markdown("#### Recommendations")
        for rec in recommendations:
            if rec.startswith('⚠️'):
                st.warning(rec)
            elif rec.startswith('✅'):
                st.success(rec)
            else:
                st.info(rec)
    
    # Show detected column mappings
    st.markdown("#### Detected Columns")
    detected_cols = preview.get('detected_columns', {})
    
    col_display = {
        'date_column': 'Date',
        'description_column': 'Description',
        'amount_column': 'Amount',
        'credit_column': 'Credit',
        'debit_column': 'Debit',
        'transaction_type_column': 'Transaction Type',
        'account_name_column': 'Account Name',
        'category_column': 'Category',
        'balance_column': 'Balance'
    }
    
    cols_data = []
    for field_name, display_name in col_display.items():
        detected_col = detected_cols.get(field_name)
        if detected_col:
            cols_data.append({
                'Field': display_name,
                'Detected Column': detected_col,
                'Status': '✅'
            })
    
    if cols_data:
        cols_df = pd.DataFrame(cols_data)
        st.dataframe(cols_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No columns detected - manual mapping required")
    
    # Show detected accounts
    account_preview = preview.get('account_preview', [])
    if account_preview:
        st.markdown("#### Detected Accounts")
        st.caption(f"Found {len(account_preview)} account(s) in this statement")
        
        for acc in account_preview:
            with st.expander(f"**{acc['account_name']}** - {acc['account_type'].replace('_', ' ').title()} ({acc['transaction_count']} transactions)", expanded=True):
                st.write(f"**Account Type:** {acc['account_type'].replace('_', ' ').title()}")
                st.write(f"**Transactions:** {acc['transaction_count']}")
                
                if acc['sample_transactions']:
                    st.write("**Sample Transactions:**")
                    for i, desc in enumerate(acc['sample_transactions'], 1):
                        st.caption(f"{i}. {desc}")
                
                # Allow user to override account type
                account_types = ['credit_card', 'checking', 'current', 'savings', 'unknown']
                current_type = acc['account_type']
                selected_type = st.selectbox(
                    f"Correct account type for '{acc['account_name']}'",
                    options=account_types,
                    index=account_types.index(current_type) if current_type in account_types else 0,
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key=f"account_type_{acc['account_name']}"
                )
    else:
        st.info("ℹ️ No separate accounts detected - all transactions will be treated as a single account")
    
    # Show sample data
    st.markdown("#### Sample Data Preview")
    st.caption("First 10 rows from your statement")
    sample_df = preview.get('sample_data')
    if sample_df is not None and not sample_df.empty:
        st.dataframe(sample_df, use_container_width=True)
    
    # Show file stats
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", preview.get('total_rows', 0))
    col2.metric("Detected Accounts", len(account_preview))
    
    # Confirmation buttons
    st.markdown("---")
    col_confirm, col_cancel = st.columns([1, 4])
    
    with col_confirm:
        if st.button("✅ Confirm & Continue", type="primary", use_container_width=True):
            st.session_state.schema_confirmed = True
            # Run Stage 2: Categorization and subscription detection
            with st.spinner("Categorizing transactions and detecting subscriptions..."):
                st.session_state.initial_result = _run_initial_categorization(
                    st.session_state.file_bytes, 
                    st.session_state.filename
                )
            st.rerun()
    
    with col_cancel:
        if st.button("❌ Cancel - Upload Different File", use_container_width=True):
            st.session_state.schema_preview = None
            st.session_state.filename = None
            st.session_state.schema_confirmed = False
            st.rerun()


def _render_subscription_confirmation(initial_result: Dict, filename: str):
    """
    Stage 2 UI: Show detected subscriptions for human confirmation before full analysis.
    This implements the HITL element for subscription validation.
    """
    st.success(f"Categorization complete for: **{filename}**")
    st.markdown("### 🔄 Subscription Detection")
    st.info("👤 **Human-In-The-Loop Check**: Please review and confirm the detected subscriptions before proceeding with expert analysis.")
    
    subscription_result = initial_result.get('subscription_result', {})
    subscriptions = subscription_result.get('subscriptions', [])
    
    if not subscriptions:
        st.warning("No subscriptions detected in this statement.")
        
        col_skip, col_back = st.columns([1, 4])
        with col_skip:
            if st.button("➡️ Skip to Analysis", type="primary", use_container_width=True):
                st.session_state.subscriptions_confirmed = True
                with st.spinner("Running full analysis..."):
                    st.session_state.pipeline_result = _run_full_analysis(
                        initial_result['transactions'],
                        []  # No confirmed subscriptions
                    )
                st.rerun()
        with col_back:
            if st.button("⬅️ Back to Schema", use_container_width=True):
                st.session_state.schema_confirmed = False
                st.session_state.initial_result = None
                st.rerun()
        return
    
    # Show statistics
    st.markdown("#### Detection Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Subscriptions", len(subscriptions))
    col2.metric("Total Monthly Cost", f"€{subscription_result.get('total_subscription_cost', 0):.2f}")
    col3.metric("Total Transactions", len(initial_result['transactions']))
    
    # Show subscription details with checkboxes
    st.markdown("#### Review Detected Subscriptions")
    st.caption("Select the subscriptions that are correct. Uncheck any false positives.")
    
    # Initialize session state for checkbox selections
    if 'subscription_selections' not in st.session_state:
        # Default: all selected
        st.session_state.subscription_selections = {sub['description']: True for sub in subscriptions}
    
    # Create DataFrame for better display
    sub_data = []
    for sub in subscriptions:
        sub_data.append({
            'Description': sub['description'],
            'Frequency': sub['frequency'],
            'Monthly Cost': f"€{sub['estimated_monthly_cost']:.2f}",
            'Occurrences': sub['occurrences'],
            'First Seen': sub['first_seen'],
            'Last Seen': sub['last_seen'],
            'LLM Validated': '✅' if sub.get('validated_by_llm') else '❌'
        })
    
    sub_df = pd.DataFrame(sub_data)
    
    # Show subscriptions with checkboxes
    st.markdown("---")
    
    confirmed_count = 0
    for idx, sub in enumerate(subscriptions):
        col_check, col_info = st.columns([1, 20])
        
        with col_check:
            is_selected = st.checkbox(
                "",
                value=st.session_state.subscription_selections[sub['description']],
                key=f"sub_check_{idx}",
                label_visibility="collapsed"
            )
            st.session_state.subscription_selections[sub['description']] = is_selected
            if is_selected:
                confirmed_count += 1
        
        with col_info:
            # Color code based on LLM validation
            validated = sub.get('validated_by_llm', False)
            color = "green" if validated else "orange"
            
            st.markdown(f"""
            <div style="padding: 10px; border-left: 3px solid {color}; background-color: rgba(0,0,0,0.05); margin-bottom: 10px;">
                <strong>{sub['description']}</strong><br/>
                <small>
                📅 {sub['frequency']} | 
                💰 €{sub['estimated_monthly_cost']:.2f}/month | 
                🔢 {sub['occurrences']} occurrences | 
                📆 {sub['first_seen']} to {sub['last_seen']} | 
                {'✅ LLM Validated' if validated else '⚠️ Pattern-based'}
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Summary of selections
    rejected_count = len(subscriptions) - confirmed_count
    col_summary1, col_summary2 = st.columns(2)
    col_summary1.metric("✅ Confirmed", confirmed_count)
    col_summary2.metric("❌ Rejected", rejected_count)
    
    # Notes field
    st.markdown("#### Additional Notes")
    review_notes = st.text_area(
        "Add any comments about the subscription detection",
        placeholder="e.g., 'Spotify is included in family plan, not a separate charge'",
        key="subscription_review_notes_input"
    )
    
    # Confirmation buttons
    st.markdown("---")
    col_confirm, col_back, col_cancel = st.columns([1, 1, 3])
    
    with col_confirm:
        if st.button("✅ Confirm & Analyze", type="primary", use_container_width=True):
            st.session_state.subscriptions_confirmed = True
            
            # Get confirmed subscriptions
            confirmed_subs = [
                sub['description'] for sub in subscriptions 
                if st.session_state.subscription_selections[sub['description']]
            ]
            
            # Store review notes in a separate variable (not the widget key)
            st.session_state.subscription_notes_saved = review_notes
            
            # Run Stage 3: Full analysis with confirmed subscriptions
            with st.spinner("Running expert analysis with confirmed subscriptions..."):
                st.session_state.pipeline_result = _run_full_analysis(
                    initial_result['transactions'],
                    confirmed_subs
                )
            st.rerun()
    
    with col_back:
        if st.button("⬅️ Back to Schema", use_container_width=True):
            st.session_state.schema_confirmed = False
            st.session_state.initial_result = None
            st.rerun()
    
    with col_cancel:
        if st.button("❌ Cancel - Start Over", use_container_width=True):
            st.session_state.schema_preview = None
            st.session_state.initial_result = None
            st.session_state.filename = None
            st.session_state.schema_confirmed = False
            st.session_state.subscriptions_confirmed = False
            st.rerun()


def _render_transactions(transactions: List[Dict]):
    st.subheader("Parsed transactions")
    if not transactions:
        st.warning("No transactions were extracted from this statement.")
        return

    tx_df = pd.DataFrame(transactions)
    tx_df['date'] = pd.to_datetime(tx_df['date']).dt.date
    
    # Check if we have account types and show summary
    if 'account_type' in tx_df.columns and tx_df['account_type'].notna().any():
        account_counts = tx_df.groupby('account_type').size()
        account_info = ', '.join([f"{acct.replace('_', ' ').title()} ({count} transactions)" for acct, count in account_counts.items()])
        st.info(f"📊 **Accounts detected:** {account_info}")
    
    st.caption(f"Showing first {min(len(tx_df), 200)} of {len(tx_df)} rows")
    st.dataframe(tx_df.head(200), use_container_width=True)


def _render_subscriptions(subscription_result: Dict):
    st.subheader("Confirmed subscriptions")
    subs = subscription_result.get('subscriptions', [])
    count = subscription_result.get('count', 0)
    monthly_cost = subscription_result.get('total_subscription_cost', 0.0)

    m1, m2 = st.columns(2)
    m1.metric("Confirmed subscriptions", count)
    m2.metric("Monthly subscription cost", f"€{monthly_cost:.2f}")
    
    if count > 0:
        st.success("✅ These subscriptions were confirmed by you during the review process.")
        
        # Show review notes if available
        if 'subscription_notes_saved' in st.session_state and st.session_state.subscription_notes_saved:
            st.info(f"**Your notes:** {st.session_state.subscription_notes_saved}")

    if not subs:
        st.info("No subscriptions were confirmed.")
        return subs

    table_cols = ['description', 'frequency', 'estimated_monthly_cost', 'occurrences', 'first_seen', 'last_seen', 'validated_by_llm']
    sub_df = pd.DataFrame(subs)[table_cols]
    sub_df.rename(columns={'estimated_monthly_cost': 'monthly_cost', 'validated_by_llm': 'llm_override'}, inplace=True)
    sub_df['monthly_cost'] = sub_df['monthly_cost'].map(lambda v: f"€{v:.2f}")
    st.dataframe(sub_df, use_container_width=True)

    return subs


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

    by_account = metrics.get('by_account', {})

    # Per-account breakdown (main view - no overall summary)
    if by_account:
        st.info("💡 Each account is analyzed separately with appropriate sign conventions. Credit cards show debt, checking/savings show available funds.")
        
        for account_type, account_metrics in by_account.items():
            st.markdown(f"### 📊 {account_type.replace('_', ' ').title()}")
            st.caption(f"{account_metrics['total_transactions']} transactions")
            
            if account_type == 'credit_card':
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total charges", f"€{account_metrics.get('total_charges', 0):.2f}", help="Purchases and charges on card")
                c2.metric("Total payments", f"€{account_metrics.get('total_payments', 0):.2f}", help="Payments made to card")
                c3.metric("Outstanding balance", f"€{account_metrics.get('net_balance', 0):.2f}", 
                         help="Current debt (positive = money owed)",
                         delta=f"€{account_metrics.get('net_balance', 0):.2f}" if account_metrics.get('net_balance', 0) > 0 else None,
                         delta_color="inverse")
                c4.metric("Payment ratio", f"{account_metrics.get('payment_ratio', 0):.1f}%", 
                         help="Payments as % of charges")
                st.caption(f"ℹ️ {account_metrics.get('interpretation', 'Positive balance = outstanding debt')}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total deposits", f"€{account_metrics.get('total_income', 0):.2f}", help="Money in")
                c2.metric("Total withdrawals", f"€{account_metrics.get('total_expenses', 0):.2f}", help="Money out")
                c3.metric("Net change", f"€{account_metrics.get('net_balance', 0):.2f}",
                         help="Change in account balance",
                         delta=f"€{account_metrics.get('net_balance', 0):.2f}",
                         delta_color="normal" if account_metrics.get('net_balance', 0) > 0 else "inverse")
                st.caption(f"ℹ️ {account_metrics.get('interpretation', 'Net change in available funds')}")
            
            st.divider()
    else:
        st.warning("No account information detected. The system works best with statements that include account identifiers.")


def _render_category_overview(category_summary: Dict, transactions: List[Dict]):
    st.subheader("Category overview")
    if not category_summary:
        st.caption("No categorized transactions yet.")
        return
    
    # Check if we have multiple accounts
    tx_df_check = pd.DataFrame(transactions) if transactions else pd.DataFrame()
    has_accounts = 'account_type' in tx_df_check.columns and tx_df_check['account_type'].notna().any()
    
    if has_accounts:
        account_types = ['All Accounts'] + sorted(tx_df_check[tx_df_check['account_type'].notna()]['account_type'].unique().tolist())
        selected_account = st.selectbox("Filter by account", account_types, key="category_account_filter")
        
        if selected_account != 'All Accounts':
            # Filter transactions by account
            transactions = [tx for tx in transactions if tx.get('account_type') == selected_account]
            st.caption(f"Showing categories for: **{selected_account.replace('_', ' ').title()}**")
            
            # Recalculate category summary for filtered transactions
            from src.agents.transaction_categorizer import TransactionCategorizerAgent
            categorizer = TransactionCategorizerAgent()
            category_summary = categorizer.get_category_summary(transactions)

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
        display_cols = ['date', 'description', 'amount']
        if has_accounts and 'account_type' in cat_df.columns:
            display_cols.append('account_type')
        st.dataframe(cat_df[display_cols], use_container_width=True)


def _render_visualizations(transactions: List[Dict], metrics: Dict):
    st.subheader("Visual insights")
    if not transactions:
        st.caption("Upload a statement to see charts.")
        return

    tx_df = pd.DataFrame(transactions).copy()
    tx_df['date'] = pd.to_datetime(tx_df['date'])
    
    # Check if we have multiple accounts
    has_accounts = 'account_type' in tx_df.columns and tx_df['account_type'].notna().any()
    
    if has_accounts:
        # Show account selector
        account_types = ['All Accounts'] + sorted(tx_df[tx_df['account_type'].notna()]['account_type'].unique().tolist())
        selected_account = st.selectbox("Filter by account", account_types, key="viz_account_filter")
        
        if selected_account != 'All Accounts':
            tx_df = tx_df[tx_df['account_type'] == selected_account]
            st.caption(f"Showing visualizations for: **{selected_account.replace('_', ' ').title()}**")

    # Enhanced color scheme
    color_scheme = px.colors.qualitative.Set2
    
    # Recalculate spending by category based on filtered transactions
    expenses_df = tx_df[tx_df['amount'] < 0].copy()
    if len(expenses_df) > 0 and 'category' in expenses_df.columns:
        expenses_df['amount_abs'] = expenses_df['amount'].abs()
        spending_by_category = expenses_df.groupby('category')['amount_abs'].sum().to_dict()
    else:
        spending_by_category = {}
    
    col1, col2 = st.columns(2)

    if spending_by_category:
        pie_fig = px.pie(
            names=list(spending_by_category.keys()),
            values=list(spending_by_category.values()),
            title="<b>Spending by Category</b>",
            color_discrete_sequence=color_scheme,
            hole=0.4  # Donut chart
        )
        pie_fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>€%{value:.2f}<br>%{percent}<extra></extra>'
        )
        pie_fig.update_layout(
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
            font=dict(size=12)
        )
        col1.plotly_chart(pie_fig, use_container_width=True)
    else:
        col1.info("No spending breakdown available.")

    # Daily cumulative balance with enhanced styling
    if has_accounts and selected_account == 'All Accounts':
        # Calculate cumulative balance by account type
        daily = tx_df.groupby(['date', 'account_type'])['amount'].sum().reset_index()
        daily = daily.sort_values(['account_type', 'date'])
        daily['cumulative_balance'] = daily.groupby('account_type')['amount'].cumsum()
        
        line_fig = px.line(
            daily, x='date', y='cumulative_balance', color='account_type',
            title='<b>Cumulative Balance by Account</b>',
            color_discrete_sequence=color_scheme
        )
        line_fig.update_traces(line_shape='spline', line=dict(width=3))
        line_fig.update_layout(
            yaxis_title="Cumulative Balance (€)",
            hovermode='x unified'
        )
    else:
        # Calculate cumulative balance for single account
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
            yaxis_title="Cumulative Balance (€)",
            hovermode='x unified'
        )
    
    line_fig.update_layout(
        xaxis_title="Date",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.5)'),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)')
    )
    line_fig.update_xaxes(showgrid=True, gridwidth=1)
    line_fig.update_yaxes(showgrid=True, gridwidth=1)
    line_fig.update_traces(hovertemplate='€%{y:.2f}<extra></extra>')
    col2.plotly_chart(line_fig, use_container_width=True)

    # Monthly summary with enhanced styling and cumulative line
    tx_df['month'] = tx_df['date'].dt.to_period('M').dt.to_timestamp()
    if has_accounts and selected_account == 'All Accounts':
        summary = tx_df.groupby(['month', 'account_type'])['amount'].sum().reset_index()
        bar_fig = px.bar(
            summary, x='month', y='amount', color='account_type',
            title='<b>Monthly Net Position by Account</b>',
            barmode='group',
            color_discrete_sequence=color_scheme
        )
        # Add cumulative lines for each account (same axis)
        for i, account_type in enumerate(summary['account_type'].unique()):
            account_data = summary[summary['account_type'] == account_type].sort_values('month')
            account_data['cumulative'] = account_data['amount'].cumsum()
            bar_fig.add_scatter(
                x=account_data['month'],
                y=account_data['cumulative'],
                mode='lines+markers',
                name=f'{account_type} (Cumulative)',
                line=dict(width=3, dash='solid'),
                marker=dict(size=8),
                hovertemplate='Cumulative: €%{y:.2f}<extra></extra>'
            )
    else:
        summary = tx_df.groupby('month')['amount'].sum().reset_index()
        summary = summary.sort_values('month')
        summary['cumulative'] = summary['amount'].cumsum()
        
        # Color bars based on positive/negative
        summary['color'] = summary['amount'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
        bar_fig = px.bar(
            summary, x='month', y='amount',
            title='<b>Monthly Net Position with Cumulative Balance</b>',
            color='color',
            color_discrete_map={'Positive': '#00CC96', 'Negative': '#EF553B'}
        )
        bar_fig.update_traces(showlegend=False, name='Monthly Change', hovertemplate='Monthly: €%{y:.2f}<extra></extra>')
        
        # Add cumulative line (same axis)
        bar_fig.add_scatter(
            x=summary['month'],
            y=summary['cumulative'],
            mode='lines+markers',
            name='Cumulative Balance',
            line=dict(color='#FFA500', width=4, dash='solid'),
            marker=dict(size=10, symbol='diamond'),
            hovertemplate='Cumulative: €%{y:.2f}<extra></extra>'
        )
    
    bar_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount (€)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.5)'),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
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

    # Initialize session state for storing results
    if 'pipeline_result' not in st.session_state:
        st.session_state.pipeline_result = None
        st.session_state.schema_preview = None
        st.session_state.initial_result = None
        st.session_state.filename = None
        st.session_state.schema_confirmed = False
        st.session_state.subscriptions_confirmed = False

    # Process new file upload - Stage 1: Schema Preview
    if uploaded_file is not None:
        if st.session_state.filename != uploaded_file.name:
            with st.spinner("Analyzing schema and detecting accounts..."):
                st.session_state.schema_preview = _get_schema_preview(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state.filename = uploaded_file.name
                st.session_state.schema_confirmed = False
                st.session_state.subscriptions_confirmed = False
                st.session_state.initial_result = None
                st.session_state.pipeline_result = None
                st.session_state.file_bytes = uploaded_file.getvalue()
    elif use_sample:
        if not SAMPLE_FILE.exists():
            st.error(f"Sample file not found at {SAMPLE_FILE}")
        elif st.session_state.filename != SAMPLE_FILE.name:
            with st.spinner("Analyzing schema and detecting accounts..."):
                st.session_state.schema_preview = _get_schema_preview(SAMPLE_FILE.read_bytes(), SAMPLE_FILE.name)
                st.session_state.filename = SAMPLE_FILE.name
                st.session_state.schema_confirmed = False
                st.session_state.subscriptions_confirmed = False
                st.session_state.initial_result = None
                st.session_state.pipeline_result = None
                st.session_state.file_bytes = SAMPLE_FILE.read_bytes()

    # Show schema confirmation UI if not yet confirmed (Stage 1)
    if st.session_state.schema_preview is not None and not st.session_state.schema_confirmed:
        _render_schema_confirmation(st.session_state.schema_preview, st.session_state.filename)
        return

    # Show subscription confirmation UI if not yet confirmed (Stage 2)
    if st.session_state.initial_result is not None and not st.session_state.subscriptions_confirmed:
        _render_subscription_confirmation(st.session_state.initial_result, st.session_state.filename)
        return

    if st.session_state.pipeline_result is None:
        st.info("Upload a bank statement to get started.")
        return
    
    pipeline_result = st.session_state.pipeline_result
    filename = st.session_state.filename

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
