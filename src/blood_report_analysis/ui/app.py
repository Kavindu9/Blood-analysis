import os
import json
import html
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Blood Work Analyzer",
    page_icon="🩸",
    layout="wide",
)

MODEL_NAME = "gemini-3.7-flash"

gemini_llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
)

openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_retries=1,
)

# Streamlit Sidebar: Provider Selection
st.sidebar.title("⚙️ AI Model Settings")
primary_provider = st.sidebar.radio(
    "Preferred Primary Model:",
    ["Gemini AI (gemini-1.5-flash)", "OpenAI (gpt-4o-mini)"]
)

# Configure primary LLM and automatic fallback chain
if "Gemini" in primary_provider:
    llm = gemini_llm.with_fallbacks([openai_llm])
    fallback_provider = "OpenAI"
else:
    llm = openai_llm.with_fallbacks([gemini_llm])
    fallback_provider = "Gemini"

st.sidebar.info(
    f"**Active Setup:** Primary is set to **{primary_provider.split()[0]}**.\n\n"
    f"If rate limits (429) or errors occur, it will automatically fall back to **{fallback_provider}**."
)



# Custom color palette for health status
COLOR_MAP = {
    "NORMAL": "#2ecc71",   # Green
    "HIGH": "#e74c3c",     # Red
    "LOW": "#3498db",      # Blue
    "UNKNOWN": "#95a5a6"   # Gray
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_response_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return str(content)


def parse_json_response(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])

    raise ValueError("Gemini did not return valid JSON.")


def normalize_status(status):
    status = str(status).upper().strip()
    if status in ["HIGH", "LOW", "NORMAL"]:
        return status
    return "UNKNOWN"


def create_dataframe(extracted_data):
    tests = extracted_data.get("tests", [])
    rows = []

    for test in tests:
        test_name = str(test.get("test_name", "Unknown Test")).strip()
        value = test.get("value")
        status = normalize_status(test.get("status", "UNKNOWN"))
        reference = str(test.get("reference_range", "Not provided")).strip()
        unit = str(test.get("unit", "")).strip()

        # Parse numeric values
        try:
            numeric_val = float(test.get("numeric_value")) if test.get("numeric_value") is not None else None
        except (TypeError, ValueError):
            numeric_val = None

        try:
            ref_min = float(test.get("ref_min")) if test.get("ref_min") is not None else None
        except (TypeError, ValueError):
            ref_min = None

        try:
            ref_max = float(test.get("ref_max")) if test.get("ref_max") is not None else None
        except (TypeError, ValueError):
            ref_max = None

        # Calculate normalized position in reference range (0% = Min, 100% = Max)
        range_pct = None
        if numeric_val is not None and ref_min is not None and ref_max is not None and ref_max > ref_min:
            range_pct = ((numeric_val - ref_min) / (ref_max - ref_min)) * 100

        rows.append({
            "Test": test_name,
            "Value": value,
            "Numeric Value": numeric_val,
            "Ref Min": ref_min,
            "Ref Max": ref_max,
            "Range Position %": range_pct,
            "Unit": unit,
            "Status": status,
            "Reference Range": reference,
        })

    return pd.DataFrame(rows)


def safe_markdown(text):
    return html.escape(str(text)).replace("\n", "<br>")


# ============================================================
# GEMINI EXTRACTION
# ============================================================

def extract_blood_data(blood_report):
    extraction_prompt = f"""
You are a medical laboratory data extraction assistant.

Extract ALL laboratory test results from the blood report below.

For every test, extract:
1. test_name: Name of the test
2. value: Original reported value text
3. numeric_value: Numeric representation (null for qualitative tests)
4. ref_min: Minimum numeric value of the normal reference range (null if unavailable)
5. ref_max: Maximum numeric value of the normal reference range (null if unavailable)
6. unit: Unit of measurement
7. status: HIGH, LOW, or NORMAL
8. reference_range: Full reference range text

Return ONLY valid JSON format:
{{
    "tests": [
        {{
            "test_name": "Hemoglobin",
            "value": "13.2",
            "numeric_value": 13.2,
            "ref_min": 13.0,
            "ref_max": 17.0,
            "unit": "g/dL",
            "status": "NORMAL",
            "reference_range": "13.0 - 17.0 g/dL"
        }}
    ]
}}

BLOOD REPORT:
{blood_report}
"""

    response = llm.invoke(extraction_prompt)
    response_text = get_response_text(response)
    return parse_json_response(response_text)


def generate_health_summary(extracted_values):
    diet_prompt = f"""
You are a clinical nutritionist familiar with Sri Lankan dietary habits.

Review the extracted blood test results below.

Provide:
SECTION 1 - HEALTH SUMMARY
Write a simple summary in 4-5 lines explaining abnormal findings without diagnosing diseases.

SECTION 2 - SRI LANKAN DIET PLAN
Provide ONLY these two subsections:
1. Foods to eat more of
2. Foods to limit or avoid

Use practical Sri Lankan foods (e.g., Red rice, Gotukola, Dhal, Fish, Coconut limitations).

Extracted Results:
{json.dumps(extracted_values, indent=2)}
"""
    response = llm.invoke(diet_prompt)
    return get_response_text(response)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def render_summary_metrics(df):
    total_tests = len(df)
    normal_count = len(df[df["Status"] == "NORMAL"])
    high_count = len(df[df["Status"] == "HIGH"])
    low_count = len(df[df["Status"] == "LOW"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tests Extracted", total_tests)
    col2.metric("Normal Range", normal_count)
    col3.metric("High Values ⚠️", high_count, delta_color="inverse")
    col4.metric("Low Values ⚠️", low_count, delta_color="inverse")


def render_pie_chart(df):
    status_counts = df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        title="Overall Test Status Distribution",
        hole=0.45,
        color="Status",
        color_discrete_map=COLOR_MAP
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}"
    )

    fig.update_layout(
        height=380,
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def render_range_position_chart(df):
    """
    Renders a normalized bar chart (0% to 100% scale).
    0% = Normal Minimum, 100% = Normal Maximum.
    Values < 0% are LOW, values > 100% are HIGH.
    """
    chart_df = df.dropna(subset=["Range Position %"]).copy()

    if chart_df.empty:
        st.info("Insufficient reference range data to compute relative normalized scale.")
        return

    # Cap visual limit for extreme outliers so the chart remains readable
    chart_df["Plot Value"] = chart_df["Range Position %"].clip(-30, 130)

    fig = px.bar(
        chart_df,
        x="Plot Value",
        y="Test",
        orientation="h",
        color="Status",
        color_discrete_map=COLOR_MAP,
        title="Relative Test Position vs. Normal Reference Range (0% = Min, 100% = Max)",
        hover_data=["Value", "Unit", "Reference Range", "Status"]
    )

    # Reference zone line markers (0% and 100%)
    fig.add_vline(x=0, line_dash="dash", line_color="#3498db", annotation_text="Min Limit")
    fig.add_vline(x=100, line_dash="dash", line_color="#e74c3c", annotation_text="Max Limit")

    fig.update_layout(
        height=max(400, len(chart_df) * 35),
        xaxis_title="Relative Percentage inside Normal Range (%)",
        yaxis_title="Test Name",
        xaxis=dict(range=[-40, 140]),
        margin=dict(t=50, b=50, l=20, r=20),
        yaxis={'categoryorder':'total ascending'}
    )

    st.plotly_chart(fig, use_container_width=True)


def render_abnormal_breakdown(df):
    abnormal_df = df[df["Status"].isin(["HIGH", "LOW"])].copy()

    if abnormal_df.empty:
        st.success("🎉 All extracted parameters are within normal reference ranges!")
        return

    st.subheader("⚠️ Critical & Abnormal Markers")
    
    fig = px.bar(
        abnormal_df,
        x="Test",
        y="Numeric Value",
        color="Status",
        color_discrete_map=COLOR_MAP,
        text="Value",
        title="Deviating Lab Parameters",
        hover_data=["Unit", "Reference Range"]
    )
    
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350, margin=dict(t=40, b=40, l=10, r=10))
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🩸 Blood Work Visual Analyzer")
st.caption("AI extraction, standardized laboratory visualization, and personalized diet plan.")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Blood Work Report Input")
    blood_report = st.text_area(
        label="Paste report text",
        height=380,
        placeholder="Paste full text of blood report here...",
        label_visibility="collapsed"
    )
    analyze_clicked = st.button("🔍 Analyze Report", type="primary", use_container_width=True)

with right_col:
    st.subheader("Clinical Summary")
    health_box = st.empty()
    health_box.info("Clinical summary will appear here after analysis.")

    st.subheader("Dietary Guidance")
    diet_box = st.empty()
    diet_box.info("Dietary recommendations will appear here after analysis.")


# ============================================================
# EXECUTION WORKFLOW
# ============================================================

if analyze_clicked:
    if not blood_report.strip():
        st.warning("Please paste a report before running analysis.")
    else:
        try:
            with st.spinner("Extracting parameters and reference boundaries..."):
                extracted_data = extract_blood_data(blood_report)
                extracted_values = extracted_data.get("tests", [])

                if not extracted_values:
                    st.error("No lab results could be extracted.")
                    st.stop()

                df = create_dataframe(extracted_data)
                st.session_state["blood_df"] = df
                st.session_state["extracted_data"] = extracted_data

            with st.spinner("Generating clinical summary & diet plan..."):
                full_response = generate_health_summary(extracted_data)

            if "SECTION 2" in full_response:
                parts = full_response.split("SECTION 2", 1)
                health_summary = parts[0].replace("SECTION 1 - HEALTH SUMMARY", "").strip()
                diet_plan = parts[1].replace("- SRI LANKAN DIET PLAN:", "").strip()
            else:
                health_summary = full_response
                diet_plan = ""

            health_box.markdown(f"**Summary:**\n{health_summary}")
            diet_box.markdown(f"{diet_plan}")
            st.success("Analysis Complete!")

        except Exception as e:
            st.error("An error occurred during analysis.")
            st.exception(e)


# ============================================================
# DASHBOARD RENDERING
# ============================================================

if "blood_df" in st.session_state:
    df = st.session_state["blood_df"]

    st.divider()
    st.header("📊 Interactive Visual Analytics Dashboard")

    render_summary_metrics(df)
    st.divider()

    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        render_pie_chart(df)

    with chart_col2:
        render_abnormal_breakdown(df)

    st.divider()
    st.subheader("📈 Normalized Reference Range Position")
    st.caption("Visualizes where each test falls relative to its normal minimum (0%) and maximum (100%).")
    render_range_position_chart(df)

    st.divider()
    st.subheader("📋 All Extracted Results")
    st.dataframe(
        df[["Test", "Value", "Unit", "Status", "Reference Range"]],
        use_container_width=True,
        hide_index=True
    )