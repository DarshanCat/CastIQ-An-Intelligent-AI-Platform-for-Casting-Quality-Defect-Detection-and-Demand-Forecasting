import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_FILE = BASE / 'backend' / 'data' / 'inventory_data.json'

def load_inventory():
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading inventory data: {e}")
        return {}

def save_inventory(data):
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving inventory data: {e}")
        return False

def render_inventory_tracker():
    st.subheader("📦 Alloy & Supply Inventory Tracker")
    st.markdown("##### Real-time raw metal supplies, alloy additive reserves, and warehouse transactions")
    st.markdown("---")
    
    inventory = load_inventory()
    if not inventory:
        return

    # Check for low stocks and display warnings
    low_stocks = []
    for item, details in inventory.items():
        if details["quantity"] < details["safety_stock"]:
            low_stocks.append(f"⚠️ **{item}** is low: **{details['quantity']}{details['unit']}** (Safety limit: {details['safety_stock']}{details['unit']})")
            
    if low_stocks:
        for alert in low_stocks:
            st.warning(alert)
            
    # Metric Stock overview columns
    items = list(inventory.keys())
    cols = st.columns(len(items))
    for i, item in enumerate(items):
        details = inventory[item]
        q = details["quantity"]
        s = details["safety_stock"]
        unit = details["unit"]
        
        status_color = "normal" if q >= s else "inverse"
        cols[i].metric(label=item, value=f"{q:,.0f} {unit}", 
                       delta=f"Safety: {s:,.0f}" if q >= s else f"Shortfall: {s-q:,.0f}", 
                       delta_color=status_color)

    st.markdown("---")
    
    # Stock Charts
    st.markdown("### 📊 Stock Levels vs Safety Limits")
    stock_rows = []
    for item, details in inventory.items():
        stock_rows.append({"Supply Item": item, "Amount": details["quantity"], "Type": "Current Inventory"})
        stock_rows.append({"Supply Item": item, "Amount": details["safety_stock"], "Type": "Safety Threshold"})
        
    df_stock = pd.DataFrame(stock_rows)
    fig = px.bar(df_stock, x="Supply Item", y="Amount", color="Type", barmode="group",
                 color_discrete_map={"Current Inventory": "#38bdf8", "Safety Threshold": "#f43f5e"})
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=20, b=0),
        height=320
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    
    # Transaction Panel
    st.markdown("### ➕ Log Warehouse Transaction")
    t1, t2, t3 = st.columns(3)
    
    with t1:
        tx_item = st.selectbox("Select Material / Supply", items)
    with t2:
        tx_action = st.radio("Action Type", ["Receive Supplies (Add)", "Disburse Material (Use)"])
    with t3:
        details = inventory[tx_item]
        tx_qty = st.number_input(f"Quantity to Log ({details['unit']})", min_value=1.0, value=100.0, step=10.0)
        
    log_btn = st.button("💾 Commit Transaction & Update Stock", width="stretch", type="primary")
    
    if log_btn:
        current_qty = details["quantity"]
        if "Receive" in tx_action:
            new_qty = current_qty + tx_qty
            action_desc = "added"
        else:
            if current_qty < tx_qty:
                st.error(f"Cannot deduct {tx_qty}{details['unit']}! Only {current_qty}{details['unit']} available in stock.")
                return
            new_qty = current_qty - tx_qty
            action_desc = "deducted"
            
        inventory[tx_item]["quantity"] = new_qty
        if save_inventory(inventory):
            st.success(f"Successfully {action_desc} {tx_qty} {details['unit']} of **{tx_item}**! New stock level: **{new_qty:,.0f} {details['unit']}**.")
            st.rerun()
