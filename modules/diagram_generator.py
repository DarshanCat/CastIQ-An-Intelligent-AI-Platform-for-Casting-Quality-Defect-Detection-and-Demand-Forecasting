import streamlit as st
import os
import re
from datetime import datetime

DIAGRAM_EXAMPLES = {
    "Casting Process Flow": "Generate a casting process flow diagram for ductile iron centrifugal casting showing steps: Raw Material → Melting → Inoculation → Centrifugal Casting → Cooling → Knockout → Shot Blasting → Inspection → Dispatch",
    "Defect Analysis": "Generate a defect analysis diagram for a ductile iron hydraulic manifold showing defect locations: porosity on inner bore, cracks near flange, shrinkage at thick sections, cold shut at parting line",
    "Mold Cross Section": "Generate a 2D mold cross section diagram for centrifugal casting showing: outer mold shell, inner cavity, metal pour point, rotation axis, solidified metal layer, and cooling zones",
    "Quality Control Flow": "Generate a quality control flowchart for VSPL casting inspection showing: Dimensional Check → Visual Inspection → NDT Testing → Hardness Test → Pass/Fail decision → Rework or Dispatch",
    "Heat Treatment Cycle": "Generate a heat treatment cycle diagram showing temperature vs time for IS 600/3 ductile iron: ramp up to 920°C, hold for 2 hours, air cool to room temperature, temper at 600°C, final cool",
    "Alloy Composition": "Generate a pie chart style composition diagram for IS 500/7 ductile iron showing: Carbon 3.5%, Silicon 2.3%, Manganese 0.3%, Magnesium 0.05%, Iron remainder",
}

SYSTEM_PROMPT = """You are an expert technical diagram generator for VSPL (Vijay Spheroidals Pvt Ltd), a ductile iron casting manufacturer.

Your job is to generate clean, professional SVG diagrams based on the user's text prompt.

STRICT RULES:
1. Return ONLY raw SVG code — no markdown, no explanation, no ```svg blocks, nothing else
2. SVG must start with <svg and end with </svg>
3. Use viewBox="0 0 800 500" always
4. Dark industrial theme: background #0e1118, shapes #111722, borders #1a2030
5. Gold accent color #c9a84c for titles, important elements, arrows
6. Text colors: titles #e8eef5, labels #c8d8e8, descriptions #4a6070
7. Use clean rectangles, circles, arrows (lines with markers), text
8. For flow diagrams: boxes connected with arrows, left to right or top to bottom
9. For defect diagrams: show a simple part outline with colored markers for defect locations
10. For cross sections: show layered rectangles/circles with labels
11. Arrow style: use <defs><marker> with gold fill for arrowheads
12. Font: font-family="Arial, sans-serif"
13. Make it look professional and technical — like an engineering drawing
14. Always add a title at the top and VSPL watermark at bottom right
15. Viewport Coordinates: Ensure all elements are fully visible and centered within the 800x500 canvas. Do not use negative coordinates or coordinates that extend beyond 800 width or 500 height.

DIAGRAM TYPES YOU SUPPORT:
- Process flow diagrams (boxes + arrows)
- Defect analysis (part outline + defect markers)  
- Mold cross sections (2D layered view)
- QC flowcharts (decision diamonds + boxes)
- Heat treatment cycles (line graph style)
- Alloy composition diagrams (pie/donut chart style)
"""

def call_gemini_svg(prompt: str) -> str:
    import requests
    import time
    api_key = st.session_state.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        return None, "No Gemini API key set"
    
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    headers = {"Content-Type": "application/json"}

    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    last_err = ""
    
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        max_retries = 2
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                
                if resp.status_code == 429:
                    last_err = f"API error 429: Quota exceeded for {model}."
                    break
                    
                if resp.status_code != 200:
                    last_err = f"API error {resp.status_code} for {model}: {resp.text[:200]}"
                    break
                    
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    fb = data.get("promptFeedback", {})
                    last_err = f"No candidates returned for {model}. promptFeedback: {fb}"
                    break
                    
                cand = candidates[0]
                parts = cand.get("content", {}).get("parts", [])
                if not parts:
                    last_err = f"No content parts for {model}. finishReason: {cand.get('finishReason')}"
                    break
                    
                svg = parts[0]["text"].strip()
                svg = re.sub(r"^```svg\s*", "", svg)
                svg = re.sub(r"^```\s*", "", svg)
                svg = re.sub(r"\s*```$", "", svg)
                svg = svg.strip()
                if not svg.startswith("<svg"):
                    idx = svg.find("<svg")
                    if idx != -1:
                        svg = svg[idx:]
                return svg, None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = f"Connection error for {model}: {e}"
                if attempt < max_retries - 1:
                    time.sleep(1.5)
                    continue
            except Exception as e:
                last_err = f"Error for {model}: {e}"
                break
                
    return None, f"All models failed. Last error: {last_err}"

def render_diagram_generator():
    st.title("✏️ AI Diagram Generator")
    st.markdown('<div style="color:#4a6070;font-size:13px;margin-bottom:20px">Type any prompt → Gemini AI generates a professional technical diagram as SVG</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.subheader("📝 Describe Your Diagram")

        # Quick examples
        st.markdown('<div style="color:#4a6070;font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:6px">QUICK EXAMPLES</div>', unsafe_allow_html=True)
        for label in DIAGRAM_EXAMPLES:
            if st.button(label, width="stretch", key=f"ex_{label}"):
                st.session_state['diagram_prompt'] = DIAGRAM_EXAMPLES[label]

        st.markdown("---")

        # Prompt input
        prompt = st.text_area(
            "Your prompt",
            value=st.session_state.get('diagram_prompt', ''),
            height=160,
            placeholder="e.g. Generate a casting process flow diagram showing melting → pouring → cooling → inspection steps for VSPL ductile iron manufacturing",
            label_visibility="collapsed"
        )

        # Options
        c1, c2 = st.columns(2)
        diagram_type = c1.selectbox("Diagram Type", [
            "Auto Detect",
            "Process Flow",
            "Defect Analysis",
            "Cross Section",
            "Flowchart",
            "Graph/Chart",
        ])
        style_hint = c2.selectbox("Style", [
            "Industrial Dark",
            "Technical Drawing",
            "Simple Clean",
        ])

        generate_btn = st.button(
            "⚡ Generate Diagram",
            width="stretch",
            type="primary",
            disabled=not prompt.strip()
        )

        # Tips
        st.markdown("""
<div style="background:#111722;border:1px solid #1a2030;border-left:3px solid #c9a84c;
            border-radius:5px;padding:12px 14px;margin-top:8px">
  <div style="color:#c9a84c;font-size:10px;font-weight:700;letter-spacing:1px;margin-bottom:6px">PROMPT TIPS</div>
  <div style="color:#4a6070;font-size:11px;line-height:1.7">
    ✓ Mention specific steps or parts<br>
    ✓ Say how many elements to show<br>
    ✓ Mention direction (left→right, top→down)<br>
    ✓ Mention part name for defect diagrams<br>
    ✓ Include values for chart diagrams
  </div>
</div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("🖼️ Generated Diagram")

        if generate_btn and prompt.strip():
            # Enhance prompt with type hint
            full_prompt = prompt
            if diagram_type != "Auto Detect":
                full_prompt = f"[{diagram_type.upper()}] {prompt}"
            if style_hint != "Industrial Dark":
                full_prompt += f" Style: {style_hint}"

            with st.spinner("🤖 Gemini is drawing your diagram..."):
                svg, error = call_gemini_svg(full_prompt)

            if error:
                if "No Gemini API key" in error:
                    st.error("⚠️ No Gemini API key set. Add it in the sidebar.")
                else:
                    st.error(f"Generation failed: {error}")
            elif svg:
                st.session_state['last_svg']    = svg
                st.session_state['last_prompt'] = prompt
                st.success("✅ Diagram generated!")

        # Display diagram
        if 'last_svg' in st.session_state:
            svg_display = st.session_state['last_svg']
            # Wrap in responsive container
            st.markdown(f"""
<div style="background:#0b0e15;border:1px solid #1a2030;border-radius:8px;
            padding:12px;overflow:hidden">
  {svg_display}
</div>""", unsafe_allow_html=True)

            st.markdown("---")
            dc1, dc2, dc3 = st.columns(3)

            # Download SVG
            dc1.download_button(
                "⬇️ Download SVG",
                data=st.session_state['last_svg'],
                file_name=f"VSPL_Diagram_{datetime.today().strftime('%Y%m%d_%H%M')}.svg",
                mime="image/svg+xml",
                width="stretch"
            )

            # Download as HTML (viewable in browser)
            html_wrap = f"""<!DOCTYPE html>
<html><head><title>VSPL Diagram</title>
<style>body{{background:#0e1118;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}}</style>
</head><body>{st.session_state['last_svg']}</body></html>"""
            dc2.download_button(
                "⬇️ Download HTML",
                data=html_wrap,
                file_name=f"VSPL_Diagram_{datetime.today().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                width="stretch"
            )

            # Regenerate
            if dc3.button("🔄 Regenerate", width="stretch", type="secondary"):
                if 'last_prompt' in st.session_state:
                    with st.spinner("Regenerating..."):
                        svg2, err2 = call_gemini_svg(st.session_state['last_prompt'])
                    if svg2:
                        st.session_state['last_svg'] = svg2
                        st.rerun()

            # Show last prompt used
            st.markdown(f'<div style="color:#2a3a4a;font-size:10px;margin-top:4px">Last prompt: {st.session_state.get("last_prompt","")[:80]}...</div>', unsafe_allow_html=True)

        else:
            # Empty state
            st.markdown("""
<div style="background:#0b0e15;border:1px dashed #1a2030;border-radius:8px;
            height:340px;display:flex;align-items:center;justify-content:center">
  <div style="text-align:center">
    <div style="font-size:48px;margin-bottom:12px">✏️</div>
    <div style="color:#2a3a4a;font-size:13px">Your diagram will appear here</div>
    <div style="color:#2a3a4a;font-size:11px;margin-top:4px">Pick a quick example or type your own prompt</div>
  </div>
</div>""", unsafe_allow_html=True)

    # History section
    st.markdown("---")
    st.subheader("📋 Diagram Types Reference")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
<div style="background:#111722;border:1px solid #1a2030;border-top:2px solid #60a5fa;border-radius:6px;padding:14px">
  <div style="color:#60a5fa;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:8px">PROCESS FLOW</div>
  <div style="color:#c8d8e8;font-size:12px;margin-bottom:4px">Best for: Manufacturing steps</div>
  <div style="color:#4a6070;font-size:11px;line-height:1.6">
    Melting → Casting → Cooling<br>
    Boxes connected by arrows<br>
    Left-to-right or top-down
  </div>
</div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("""
<div style="background:#111722;border:1px solid #1a2030;border-top:2px solid #f87171;border-radius:6px;padding:14px">
  <div style="color:#f87171;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:8px">DEFECT ANALYSIS</div>
  <div style="color:#c8d8e8;font-size:12px;margin-bottom:4px">Best for: QC reports</div>
  <div style="color:#4a6070;font-size:11px;line-height:1.6">
    Part outline with markers<br>
    Color-coded by severity<br>
    Location + defect type labels
  </div>
</div>""", unsafe_allow_html=True)

    with r3:
        st.markdown("""
<div style="background:#111722;border:1px solid #1a2030;border-top:2px solid #34d399;border-radius:6px;padding:14px">
  <div style="color:#34d399;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:8px">CROSS SECTION</div>
  <div style="color:#c8d8e8;font-size:12px;margin-bottom:4px">Best for: Mold design</div>
  <div style="color:#4a6070;font-size:11px;line-height:1.6">
    2D layered view of mold<br>
    Material zones labeled<br>
    Dimensions and annotations
  </div>
</div>""", unsafe_allow_html=True)