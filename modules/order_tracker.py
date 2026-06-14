import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_FILE = BASE / 'backend' / 'data' / 'orders_data.json'

STAGES = ["Melt", "Cast", "Heat Treat", "Machining", "QC", "Dispatch"]

def load_orders():
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading orders data: {e}")
        return []

def render_order_tracker():
    st.subheader("🚚 Manufacturing Order Tracker")
    st.markdown("##### Centrifugal casting order status pipeline, mechanical specifications, and manufacturing milestones")
    st.markdown("---")
    
    orders = load_orders()
    if not orders:
        return

    # Visual pipeline metrics
    total_qty = sum(o["qty_kg"] for o in orders)
    active_count = sum(1 for o in orders if o["status"] == "In Progress")
    scheduled_count = sum(1 for o in orders if o["status"] == "Scheduled")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Order Backlog", f"{total_qty:,.0f} kg", delta="+4,200 kg")
    c2.metric("Orders In Production", f"{active_count} Active")
    c3.metric("Scheduled for Melt", f"{scheduled_count} Queued")

    st.markdown("---")
    
    # Progress pipelines chart
    st.markdown("### 📊 Order Manufacturing Progress Stages")
    
    order_rows = []
    for o in orders:
        current_stage = o["stage"]
        stage_idx = STAGES.index(current_stage) + 1
        pct_complete = (stage_idx / len(STAGES)) * 100.0
        order_rows.append({
            "Order ID": o["order_id"],
            "Customer": o["customer"],
            "Casting Grade": o["grade"],
            "Completed Stages": stage_idx,
            "Total Stages": len(STAGES),
            "Progress (%)": pct_complete,
            "Current Stage": current_stage
        })
        
    df_orders = pd.DataFrame(order_rows)
    
    # Renders a sleek horizontal progress bar chart
    fig = px.bar(df_orders, x="Progress (%)", y="Order ID", orientation="h",
                 hover_data=["Customer", "Casting Grade", "Current Stage"],
                 color="Current Stage", text="Current Stage",
                 color_discrete_sequence=px.colors.sequential.Blues)
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=20, b=0),
        height=320
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    
    # Order specifications table
    st.markdown("### 📋 Manufacturing Pipeline Details")
    styled_rows = []
    for o in orders:
        status_emoji = "🔵" if o["status"] == "In Progress" else "🟡"
        styled_rows.append({
            "Order ID": f"**{o['order_id']}**",
            "Customer Name": o["customer"],
            "Alloy Grade": o["grade"],
            "Order Qty (kg)": f"{o['qty_kg']:,} kg",
            "Target Due Date": o["due_date"],
            "Active Processing Stage": f"🏗️ {o['stage']}",
            "Ladle Status": f"{status_emoji} {o['status']}"
        })
        
    st.dataframe(pd.DataFrame(styled_rows), width="stretch", hide_index=True)
