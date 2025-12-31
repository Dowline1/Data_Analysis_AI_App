"""
LangGraph-based Streamlit Application

This is the updated main application that uses the LangGraph workflow
instead of direct agent calls. Supports HITL checkpoints and streaming.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
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

from src.graph.workflow import create_analysis_workflow, create_initial_state
from src.graph.state import AnalysisState
from src.agents.chat_with_data_agent import ChatWithDataAgent

# Initialize workflow (cached)
@st.cache_resource(show_spinner=False)
def get_workflow():
    """Get the compiled LangGraph workflow."""
    return create_analysis_workflow()


@st.cache_resource(show_spinner=False)
def get_chat_agent() -> ChatWithDataAgent:
    """Get chat agent for Q&A."""
    return ChatWithDataAgent()


def initialize_session_state():
    """Initialize Streamlit session state for HITL workflow."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    if "current_state" not in st.session_state:
        st.session_state.current_state = None
    
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    
    if "uploaded_file_content" not in st.session_state:
        st.session_state.uploaded_file_content = None


def run_workflow_until_interrupt(workflow, state: AnalysisState, config: Dict):
    """
    Run the workflow until it hits an interrupt (HITL checkpoint) or completes.
    
    Returns the final state.
    """
    try:
        # Stream events from the workflow
        for event in workflow.stream(state, config, stream_mode="values"):
            # Update current state
            st.session_state.current_state = event
            
            # Check if we hit an interrupt
            if event.get("awaiting_user_input"):
                return event
        
        # If we get here, workflow completed
        st.session_state.analysis_complete = True
        return st.session_state.current_state
        
    except Exception as e:
        st.error(f"Workflow error: {str(e)}")
        return state


def render_schema_confirmation():
    """Render UI for schema confirmation (HITL Stage 1)."""
    state = st.session_state.current_state
    schema_info = state.get("schema_info")
    
    if not schema_info:
        st.warning("No schema information available")
        return
    
    st.subheader("📋 Schema Detection Results")
    
    # Show detected columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Detected Columns:**")
        for col_type in ["date_column", "description_column", "amount_column", 
                        "account_column", "balance_column"]:
            col_name = schema_info.get(col_type)
            if col_name:
                st.success(f"✓ {col_type.replace('_', ' ').title()}: {col_name}")
            else:
                st.warning(f"⚠ {col_type.replace('_', ' ').title()}: Not detected")
    
    with col2:
        # Show warnings and recommendations
        warnings = schema_info.get("warnings", [])
        recommendations = schema_info.get("recommendations", [])
        
        if warnings:
            st.write("**⚠ Warnings:**")
            for warning in warnings:
                st.warning(warning)
        
        if recommendations:
            st.write("**💡 Recommendations:**")
            for rec in recommendations:
                st.info(rec)
    
    # Show detected accounts
    st.write("**Detected Accounts:**")
    accounts = schema_info.get("accounts", [])
    
    if accounts:
        schema_overrides = {}
        
        for i, acc in enumerate(accounts):
            col1, col2, col3 = st.columns([3, 2, 2])
            
            with col1:
                st.write(f"**{acc['account_name']}**")
                st.caption(f"{acc['transaction_count']} transactions")
            
            with col2:
                # Show detected type
                current_type = acc['account_type']
                type_emoji = {
                    "credit_card": "💳",
                    "checking": "🏦",
                    "current": "🏦",
                    "savings": "💰",
                    "unknown": "❓"
                }.get(current_type, "")
                
                st.write(f"{type_emoji} {current_type.replace('_', ' ').title()}")
            
            with col3:
                # Allow override
                new_type = st.selectbox(
                    "Override",
                    ["credit_card", "checking", "current", "savings"],
                    index=["credit_card", "checking", "current", "savings"].index(current_type) 
                          if current_type != "unknown" else 0,
                    key=f"account_type_{i}"
                )
                
                if new_type != current_type:
                    schema_overrides[acc['account_name']] = new_type
        
        # Confirmation button
        st.divider()
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("✅ Confirm Schema", type="primary", use_container_width=True):
                # Update state with overrides
                workflow = get_workflow()
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Apply overrides to state
                updated_state = dict(st.session_state.current_state)
                updated_state["schema_overrides"] = schema_overrides
                updated_state["awaiting_user_input"] = False
                updated_state["schema_confirmed"] = True
                
                # Continue workflow
                with st.spinner("Processing transactions..."):
                    final_state = run_workflow_until_interrupt(workflow, updated_state, config)
                    st.session_state.current_state = final_state
                st.rerun()
        
        with col2:
            st.caption(f"{len(schema_overrides)} overrides will be applied" if schema_overrides else "No overrides")
    else:
        st.error("No accounts detected. Please check your file format.")


def render_subscription_confirmation():
    """Render UI for subscription confirmation (HITL Stage 2)."""
    state = st.session_state.current_state
    subscriptions = state.get("detected_subscriptions", [])
    
    if not subscriptions:
        st.info("No subscriptions detected")
        
        # Button to skip this stage
        if st.button("Continue to Analysis", type="primary"):
            workflow = get_workflow()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            updated_state = dict(st.session_state.current_state)
            updated_state["awaiting_user_input"] = False
            updated_state["subscriptions_confirmed"] = True
            
            with st.spinner("Calculating metrics..."):
                final_state = run_workflow_until_interrupt(workflow, updated_state, config)
                st.session_state.current_state = final_state
            st.rerun()
        return
    
    st.subheader("🔄 Confirm Recurring Subscriptions")
    st.write(f"We detected {len(subscriptions)} potential subscriptions. Please review and confirm:")
    
    # Track selections
    selections = {}
    notes = {}
    
    for i, sub in enumerate(subscriptions):
        with st.expander(f"💳 {sub['merchant']} - ${sub['amount']:.2f} ({sub['frequency']})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Category:** {sub['category']}")
                st.write(f"**Frequency:** {sub['frequency']}")
                st.write(f"**Amount:** ${sub['amount']:.2f}")
                st.write(f"**Occurrences:** {len(sub['transaction_dates'])}")
                
                # Show transaction dates
                if sub['transaction_dates']:
                    st.caption(f"Dates: {', '.join(sub['transaction_dates'][:5])}")
                
                # Note field
                note = st.text_input(
                    "Add note (optional)",
                    key=f"sub_note_{i}",
                    placeholder="e.g., Cancelled in March"
                )
                if note:
                    notes[sub['merchant']] = note
            
            with col2:
                confirmed = st.checkbox(
                    "Confirm",
                    value=True,
                    key=f"sub_confirm_{i}"
                )
                selections[sub['merchant']] = confirmed
    
    # Confirmation button
    st.divider()
    confirmed_count = sum(1 for v in selections.values() if v)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("✅ Confirm Subscriptions", type="primary", use_container_width=True):
            workflow = get_workflow()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Update state with selections
            updated_state = dict(st.session_state.current_state)
            updated_state["subscription_selections"] = selections
            updated_state["subscription_notes"] = notes
            updated_state["awaiting_user_input"] = False
            updated_state["subscriptions_confirmed"] = True
            
            with st.spinner("Calculating metrics and generating insights..."):
                final_state = run_workflow_until_interrupt(workflow, updated_state, config)
                st.session_state.current_state = final_state
            st.rerun()
    
    with col2:
        st.caption(f"{confirmed_count} of {len(subscriptions)} subscriptions confirmed")


def render_analysis_results():
    """Render the final analysis results."""
    state = st.session_state.current_state
    
    # Show metrics
    st.subheader("📊 Financial Analysis")
    
    # Health Score
    health_score = state.get("health_score")
    if health_score:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Score",
                f"{health_score['overall_score']}/100",
                delta=None
            )
        
        with col2:
            st.metric(
                "Assessment",
                health_score['assessment'],
                delta=None
            )
        
        with col3:
            st.metric(
                "Checking Score",
                f"{health_score['checking_score']}/60",
                delta=None
            )
        
        with col4:
            st.metric(
                "Credit Card Score",
                f"{health_score['credit_card_score']}/40",
                delta=None
            )
    
    # Account Metrics
    account_metrics = state.get("account_metrics", [])
    if account_metrics:
        st.write("**Account Breakdown:**")
        
        for acc in account_metrics:
            with st.expander(f"📈 {acc['account_type'].title()} Account"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Income", f"${acc['total_income']:,.2f}")
                
                with col2:
                    st.metric("Total Expenses", f"${acc['total_expenses']:,.2f}")
                
                with col3:
                    st.metric("Net Cash Flow", f"${acc['net_cash_flow']:,.2f}")
                
                # Credit card specific
                if acc.get("total_charges"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Charges", f"${acc['total_charges']:,.2f}")
                        st.metric("Payments", f"${acc['total_payments']:,.2f}")
                    with col2:
                        st.metric("Payment Ratio", f"{acc['payment_ratio']:.1%}")
                        st.metric("Net Balance", f"${acc['net_balance']:,.2f}")
    
    # Expert Insights
    expert_report = state.get("expert_report")
    if expert_report:
        st.subheader("🧠 Expert Analysis")
        st.markdown(expert_report)
    
    # Subscriptions
    subscriptions = state.get("detected_subscriptions", [])
    confirmed_subs = [s for s in subscriptions if s.get("confirmed")]
    
    if confirmed_subs:
        st.subheader("🔄 Active Subscriptions")
        
        total_monthly = sum(s['amount'] for s in confirmed_subs if 'month' in s['frequency'].lower())
        st.metric("Total Monthly Subscriptions", f"${total_monthly:,.2f}")
        
        for sub in confirmed_subs:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{sub['merchant']}**")
                if sub.get('note'):
                    st.caption(sub['note'])
            with col2:
                st.write(sub['frequency'])
            with col3:
                st.write(f"${sub['amount']:.2f}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Bank Statement Analyzer",
        page_icon="💰",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("💰 Bank Statement Analyzer with LangGraph")
    st.caption("Powered by LangGraph, LangChain, and Google Gemini")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your bank statement (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file with your bank transactions"
    )
    
    if uploaded_file:
        # Save file content
        file_bytes = uploaded_file.read()
        st.session_state.uploaded_file_content = file_bytes
        
        # Create temporary file
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            # Start or continue workflow
            if st.session_state.current_state is None:
                # Initialize workflow
                initial_state = create_initial_state(tmp_path)
                workflow = get_workflow()
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                with st.spinner("Analyzing schema..."):
                    final_state = run_workflow_until_interrupt(workflow, initial_state, config)
                    st.session_state.current_state = final_state
            
            # Render appropriate stage
            state = st.session_state.current_state
            
            if not state:
                st.warning("No state available")
                return
            
            # Check current stage
            current_stage = state.get("current_stage")
            awaiting_input = state.get("awaiting_user_input", False)
            
            if awaiting_input and current_stage == "schema_confirmation":
                render_schema_confirmation()
            elif awaiting_input and current_stage == "subscription_confirmation":
                render_subscription_confirmation()
            elif st.session_state.analysis_complete:
                render_analysis_results()
                
                # Chat interface
                st.divider()
                st.subheader("💬 Chat with Your Data")
                
                user_question = st.chat_input("Ask a question about your finances...")
                if user_question:
                    agent = get_chat_agent()
                    
                    with st.chat_message("user"):
                        st.write(user_question)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = agent.answer_question(
                                transactions=state.get("transactions", []),
                                metrics={"account_metrics": state.get("account_metrics", [])},
                                question=user_question
                            )
                            st.write(response)
            else:
                st.info("Processing your data... Please wait.")
        
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    else:
        st.info("👆 Upload a bank statement to begin analysis")


if __name__ == "__main__":
    main()
