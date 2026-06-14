import streamlit as st
import pandas as pd
import numpy as np

def render_cost_estimation():
    st.subheader("💰 Commercial Casting Cost & Weight Estimator")
    st.markdown("##### Calculate centrifugal pipe weight from 3D geometry and generate commercial ex-works cost estimates")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Casting Geometry (centrifugal pipe)")
        od = st.slider("Outer Diameter (OD) (mm)", 50, 1000, 300, step=10)
        thickness = st.slider("Wall Thickness (mm)", 5, 100, 25, step=5)
        length = st.slider("Casting Length (mm)", 100, 6000, 2000, step=100)
        pieces = st.number_input("Number of Pieces Required", min_value=1, value=10, step=1)
        
        st.markdown("---")
        st.subheader("🪙 Pricing Rate Factors")
        base_iron = st.number_input("Base Pig Iron cost (Rs./kg)", min_value=10.0, value=48.0, step=1.0)
        alloy_add = st.number_input("Alloy Addition cost (Rs./kg)", min_value=5.0, value=12.0, step=1.0)
        labor_mach = st.number_input("Machining & Labor cost (Rs./kg)", min_value=5.0, value=15.0, step=1.0)
        markup_pct = st.slider("Target Operating Margin (%)", 5, 50, 15, step=1)

    with col2:
        st.subheader("📊 Geometric & Physical Verdict")
        
        # Centrifugal cylinder volume = pi * (R^2 - r^2) * L
        # R = OD / 2, r = R - Thickness
        R_m = (od / 2.0) / 1000.0
        r_m = (R_m - (thickness / 1000.0))
        length_m = length / 1000.0
        
        if r_m < 0:
            st.error("Wall thickness cannot exceed outer radius!")
            return
            
        vol_single = np.pi * (R_m**2 - r_m**2) * length_m
        
        # Ductile iron density is 7,100 kg/m3
        density = 7100.0
        wt_single = vol_single * density
        wt_total = wt_single * pieces
        
        # Commercial Cost Breakdown
        iron_cost = wt_total * base_iron
        alloy_cost = wt_total * alloy_add
        energy_cost = wt_total * 8.5  # Rs. 8.5 per kg for melting electricity
        labor_cost = wt_total * labor_mach
        mold_wear_cost = wt_total * 4.0  # Rs. 4 per kg for spinner mold wear
        
        subtotal = iron_cost + alloy_cost + energy_cost + labor_cost + mold_wear_cost
        markup_val = subtotal * (markup_pct / 100.0)
        total_price = subtotal + markup_val
        
        c_wt1, c_wt2 = st.columns(2)
        c_wt1.metric("Single Pipe Weight", f"{wt_single:.2f} kg", delta="Centrifugal shape")
        c_wt2.metric("Total Batch Weight", f"{wt_total:,.2f} kg", delta=f"{pieces} castings")
        
        st.markdown("---")
        st.markdown("### 📋 Commercial ex-works Cost Breakdown")
        
        cost_df = pd.DataFrame([
            {"Cost Component": "Base Pig Iron Charge", "Amount (Rs.)": f"Rs. {int(iron_cost):,}"},
            {"Cost Component": "Alloy Additive Charge", "Amount (Rs.)": f"Rs. {int(alloy_cost):,}"},
            {"Cost Component": "Melting Energy Charge", "Amount (Rs.)": f"Rs. {int(energy_cost):,}"},
            {"Cost Component": "Machining & Labor Charge", "Amount (Rs.)": f"Rs. {int(labor_cost):,}"},
            {"Cost Component": "Spinner Mold Wear Reserve", "Amount (Rs.)": f"Rs. {int(mold_wear_cost):,}"},
            {"Cost Component": "**Production Subtotal**", "Amount (Rs.)": f"Rs. {int(subtotal):,}"},
            {"Cost Component": f"**Operating Markup ({markup_pct}%)**", "Amount (Rs.)": f"Rs. {int(markup_val):,}"},
            {"Cost Component": "🏆 **Total ex-works Quote Price**", "Amount (Rs.)": f"**Rs. {int(total_price):,}**"},
            {"Cost Component": "💡 Unit Rate per Casting", "Amount (Rs.)": f"Rs. {int(total_price / pieces):,}"}
        ])
        
        st.dataframe(cost_df, width="stretch", hide_index=True)
        
        # Download Quote Text
        quote_text = f"""==================================================
  VIJAY SPHEROIDALS PVT LTD
  Centrifugal Casting Quotation
==================================================

  Geometry Specifications:
  - Outer Diameter (OD) : {od} mm
  - Wall Thickness      : {thickness} mm
  - Casting Length      : {length} mm
  - Number of Castings  : {pieces} pcs

  Weight Calculations:
  - Single Pipe Weight  : {wt_single:.2f} kg
  - Total Batch Weight  : {wt_total:,.2f} kg

  Commercial Pricing Breakdown (ex-works):
  - Raw Pig Iron cost   : Rs. {int(iron_cost):,}
  - Alloy Additive cost : Rs. {int(alloy_cost):,}
  - Melting Energy cost : Rs. {int(energy_cost):,}
  - Labor & Machining   : Rs. {int(labor_cost):,}
  - Mold Wear Reserve   : Rs. {int(mold_wear_cost):,}
  - Operating Margin    : Rs. {int(markup_val):,} ({markup_pct}%)
  ------------------------------------------------
  TOTAL QUOTE PRICE     : Rs. {int(total_price):,} ex-works
  UNIT RATE per Pipe    : Rs. {int(total_price / pieces):,} ex-works

==================================================
  www.vijayspheroidals.in | Peenya, Bengaluru
=================================================="""

        st.download_button("⬇️ Download Commercial Quotation (TXT)", quote_text,
                           file_name=f"VSPL_Quotation_{int(wt_total)}kg.txt",
                           mime="text/plain", width="stretch")
