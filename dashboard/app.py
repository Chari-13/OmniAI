"""
Streamlit Dual-View Dashboard -- Customer Review Intelligence Platform.

Two views:
  1. Customer View ("The Real Story") -- Simplified honesty metrics
  2. Company View ("Technical Deep-Dive") -- Full analytics + roadmap

Usage:
  streamlit run dashboard/app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -- Config --------------------------------------------------------------------

API_BASE = "http://localhost:8000"
st.set_page_config(
    page_title="Omni-AI Review Intelligence",
    page_icon="🔮",  
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ----------------------------------------------------------------

st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0F0A1E 0%, #1A1145 50%, #0D0B2E 100%);
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 0.5rem 0;
    }

    .sub-header {
        color: #A5B4FC;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #A5B4FC;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 0.3rem;
    }

    /* View toggle */
    .view-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .badge-customer {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-company {
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Anomaly severity badges */
    .severity-critical { color: #EF4444; font-weight: 700; }
    .severity-high { color: #F97316; font-weight: 600; }
    .severity-medium { color: #EAB308; }
    .severity-low { color: #22C55E; }

    /* Sarcasm badge */
    .sarcasm-badge {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# -- API Helpers ---------------------------------------------------------------

def api_get(endpoint: str) -> dict:
    """GET request to the backend API."""
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Start it with: `python main.py`")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def api_post(endpoint: str, data: dict = None, files: dict = None) -> dict:
    """POST request to the backend API."""
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=data, files=files, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Start it with: `python main.py`")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


# -- Sidebar -------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with navigation and data upload."""
    with st.sidebar:
        st.markdown('<div class="main-header" style="font-size:1.5rem;">Omni-AI</div>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header" style="font-size:0.8rem;">Review Intelligence Platform</p>', unsafe_allow_html=True)

        # View selector
        view = st.radio(
            "Dashboard View",
            ["Customer View", "Company View"],
            index=0,
            help="Switch between simplified customer view and technical deep-dive",
        )

        st.divider()

        # Data upload section
        st.subheader("Data Ingestion")

        upload_type = st.selectbox("Upload Type", ["CSV File", "JSON File", "Single Review"])

        if upload_type == "CSV File":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
            skip_clean = st.checkbox("Skip text cleaning (faster)", value=True)
            if uploaded_file and st.button("Process CSV", type="primary", use_container_width=True):
                with st.spinner("Processing reviews..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    data = {"skip_cleaning": str(skip_clean).lower(), "run_anomaly_detection": "true", "generate_roadmap": "true"}
                    try:
                        resp = requests.post(
                            f"{API_BASE}/ingest/csv",
                            files=files,
                            data=data,
                            timeout=300,
                        )
                        result = resp.json()
                        if resp.status_code == 200:
                            st.success(f"Processed {result.get('successful', 0)} reviews!")
                            st.rerun()
                        else:
                            st.error(f"Error: {result.get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

        elif upload_type == "JSON File":
            uploaded_file = st.file_uploader("Upload JSON", type=["json"], key="json_upload")
            if uploaded_file and st.button("Process JSON", type="primary", use_container_width=True):
                with st.spinner("Processing reviews..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
                    try:
                        resp = requests.post(f"{API_BASE}/ingest/json", files=files, timeout=300)
                        result = resp.json()
                        if resp.status_code == 200:
                            st.success(f"Processed {result.get('successful', 0)} reviews!")
                            st.rerun()
                        else:
                            st.error(f"Error: {result.get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

        elif upload_type == "Single Review":
            review_text = st.text_area("Review Text", height=100)
            category = st.selectbox("Category", ["electronics", "food", "services", "general"])
            if review_text and st.button("Analyze", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result = api_post("/ingest", {"text": review_text, "category": category})
                    if result:
                        st.success("Analysis complete!")
                        st.json(result)

        st.divider()

        # System status
        status = api_get("/status")
        if status:
            st.caption(f"Reviews: {status.get('total_processed', 0)}")
            st.caption(f"LLM: {'Online' if status.get('llm_available') else 'Offline'}")
            st.caption(f"Engine: v{status.get('uptime_seconds', 0):.0f}s uptime")

        # Export
        st.divider()
        if st.button("Download PDF Report", use_container_width=True):
            try:
                resp = requests.get(f"{API_BASE}/report/pdf", timeout=60)
                if resp.status_code == 200:
                    st.download_button(
                        "Save PDF",
                        data=resp.content,
                        file_name=f"intelligence_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.warning("Process some reviews first!")
            except Exception:
                st.warning("Backend not available for PDF export.")

        # Reset
        if st.button("Reset All Data", type="secondary", use_container_width=True):
            api_get_raw = requests.delete(f"{API_BASE}/data/reset", timeout=10)
            st.rerun()

    return view


# -- Customer View -------------------------------------------------------------

def render_customer_view():
    """Render the Customer View ('The Real Story')."""
    st.markdown('<div class="main-header">The Real Story</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">What customers actually think -- powered by AI</p>', unsafe_allow_html=True)

    data = api_get("/dashboard/customer")
    if not data or data.get("total_reviews", 0) == 0:
        st.info("No reviews processed yet. Upload data using the sidebar to get started.")
        return

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data.get('honesty_score', 0):.0f}%</div>
            <div class="metric-label">Honesty Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data.get('total_reviews', 0)}</div>
            <div class="metric-label">Reviews Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        pos = data.get("sentiment_summary", {}).get("positive", 0)
        total = data.get("total_reviews", 1)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #10B981;">{pos/max(total,1)*100:.0f}%</div>
            <div class="metric-label">Positive Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #F59E0B;">{data.get('sarcasm_alert_count', 0)}</div>
            <div class="metric-label">Sarcasm Alerts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Strengths & Weaknesses
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top Strengths")
        for item in data.get("top_strengths", []):
            st.markdown(f"**{item['aspect']}** -- {item['score']}% positive ({item['mentions']} mentions)")
            st.progress(item["score"] / 100)

    with col_right:
        st.subheader("Top Concerns")
        for item in data.get("top_weaknesses", []):
            st.markdown(f"**{item['aspect']}** -- {item['score']}% negative ({item['mentions']} mentions)")
            st.progress(item["score"] / 100)

    st.markdown("---")

    # Sentiment pie chart
    col_chart, col_lang = st.columns(2)

    with col_chart:
        st.subheader("Overall Sentiment")
        sentiment = data.get("sentiment_summary", {})
        if sentiment:
            colors_map = {"positive": "#10B981", "negative": "#EF4444", "neutral": "#8B5CF6", "mixed": "#F59E0B"}
            fig = px.pie(
                names=[k.capitalize() for k in sentiment.keys()],
                values=list(sentiment.values()),
                color_discrete_sequence=[colors_map.get(k, "#94A3B8") for k in sentiment.keys()],
                hole=0.4,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E1",
                showlegend=True,
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_lang:
        st.subheader("Languages Detected")
        lang = data.get("language_distribution", {})
        if lang:
            fig = px.bar(
                x=list(lang.keys()),
                y=list(lang.values()),
                color_discrete_sequence=["#6366F1"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E1",
                xaxis_title="",
                yaxis_title="Count",
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Curated highlights
    st.markdown("---")
    st.subheader("What Real Customers Say")
    highlights = data.get("curated_highlights", [])
    for h in highlights:
        sentiment_color = {"positive": "#10B981", "negative": "#EF4444", "neutral": "#8B5CF6"}.get(h.get("sentiment", ""), "#94A3B8")
        sarcasm_tag = ' <span class="sarcasm-badge">Sarcasm Detected</span>' if h.get("is_sarcastic") else ""
        st.markdown(f"""
        > *"{h.get('text', '')}"*
        >
        > <span style="color:{sentiment_color}; font-weight:600">{h.get('sentiment', '').upper()}</span>
        > | Confidence: {h.get('confidence', 0):.0f}%
        > | {h.get('language', 'English')}{sarcasm_tag}
        """, unsafe_allow_html=True)


# -- Company View --------------------------------------------------------------

def render_company_view():
    """Render the Company View (Technical Deep-Dive)."""
    st.markdown('<div class="main-header">Intelligence Command Center</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Technical deep-dive with anomaly detection & strategic roadmap</p>', unsafe_allow_html=True)

    data = api_get("/dashboard/company")
    if not data or data.get("total_reviews", 0) == 0:
        st.info("No reviews processed yet. Upload data using the sidebar to get started.")
        return

    # Top metrics row
    stats = data.get("processing_stats", {})
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Reviews", data.get("total_reviews", 0))
    with col2:
        st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.1f}%")
    with col3:
        st.metric("Sarcasm Rate", f"{stats.get('sarcasm_rate', 0)*100:.1f}%")
    with col4:
        anomalies = data.get("anomaly_timeline", [])
        st.metric("Anomalies", len(anomalies))
    with col5:
        issues = data.get("issue_classifications", [])
        systemic = sum(1 for i in issues if i.get("issue_type") == "systemic")
        st.metric("Systemic Issues", systemic)

    st.markdown("---")

    # Sentiment distribution + Aspect heatmap
    col_sent, col_asp = st.columns(2)

    with col_sent:
        st.subheader("Sentiment Distribution")
        sentiment = data.get("sentiment_distribution", {})
        if sentiment:
            colors_map = {"positive": "#10B981", "negative": "#EF4444", "neutral": "#8B5CF6", "mixed": "#F59E0B"}
            fig = go.Figure(data=[go.Bar(
                x=[k.capitalize() for k in sentiment.keys()],
                y=list(sentiment.values()),
                marker_color=[colors_map.get(k, "#94A3B8") for k in sentiment.keys()],
                text=list(sentiment.values()),
                textposition="auto",
            )])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E1",
                height=300,
                margin=dict(t=20, b=40, l=40, r=20),
                yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_asp:
        st.subheader("Aspect Sentiment Heatmap")
        heatmap = data.get("aspect_heatmap", [])
        if heatmap:
            df = pd.DataFrame(heatmap)
            if not df.empty and "aspect" in df.columns:
                fig = go.Figure()
                for sentiment_type, color in [("positive", "#10B981"), ("negative", "#EF4444"), ("neutral", "#8B5CF6")]:
                    if sentiment_type in df.columns:
                        fig.add_trace(go.Bar(
                            name=sentiment_type.capitalize(),
                            x=df["aspect"],
                            y=df[sentiment_type],
                            marker_color=color,
                        ))
                fig.update_layout(
                    barmode="stack",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#CBD5E1",
                    height=300,
                    margin=dict(t=20, b=40, l=40, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Anomaly Timeline
    if anomalies:
        st.subheader("Anomaly Alerts")
        for a in anomalies:
            severity = a.get("severity", "MEDIUM")
            severity_class = f"severity-{severity.lower()}"
            st.markdown(f"""
            <div class="metric-card" style="text-align: left; margin-bottom: 0.5rem; padding: 1rem;">
                <span class="{severity_class}">[{severity}]</span>
                <strong>{a.get('type', '').replace('_', ' ').title()}</strong>
                {f" -- {a.get('aspect', '')}" if a.get('aspect') else ""}
                <br/>
                <span style="color: #94A3B8; font-size: 0.85rem;">{a.get('description', '')}</span>
                <br/>
                <span style="color: #64748B; font-size: 0.75rem;">Spike: {a.get('spike_percentage', 0):.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Issue Classifications
    if issues:
        st.subheader("Issue Classifications: Isolated vs Systemic")
        issue_df = pd.DataFrame(issues)
        if not issue_df.empty:
            display_cols = ["aspect", "issue_type", "mention_count", "total_mentions", "negative_ratio", "trend"]
            available_cols = [c for c in display_cols if c in issue_df.columns]
            st.dataframe(
                issue_df[available_cols],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # Strategic Roadmap
    roadmap = data.get("strategic_roadmap")
    if roadmap:
        st.subheader("3-Month Strategic Roadmap")

        if roadmap.get("analysis_summary"):
            st.info(roadmap["analysis_summary"])

        for month_num in [1, 2, 3]:
            theme_key = f"month_{month_num}_theme"
            theme = roadmap.get(theme_key, f"Month {month_num}")
            actions = [a for a in roadmap.get("actions", []) if a.get("month") == month_num]

            with st.expander(f"Month {month_num}: {theme}", expanded=(month_num == 1)):
                if actions:
                    for action in actions:
                        priority = action.get("priority", "MEDIUM")
                        severity_class = f"severity-{priority.lower()}"
                        st.markdown(f"""
                        **{action.get('title', 'Action')}**
                        <span class="{severity_class}">[{priority}]</span>

                        {action.get('description', '')}

                        - **Impact:** {action.get('estimated_impact', 'N/A')}
                        - **KPI:** {action.get('kpi', 'N/A')}
                        - **Owner:** {action.get('owner_suggestion', 'TBD')}

                        ---
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No actions defined for this month.")

    # Hype vs Reality (if available)
    hype = data.get("hype_vs_reality")
    if hype:
        st.markdown("---")
        st.subheader("Social Sync: Hype vs Reality")
        st.metric("Overall Hype Score", f"{hype.get('overall_hype_score', 0):.0f}/100")
        st.markdown(f"**Verdict:** {hype.get('verdict', 'N/A')}")


# -- Main App ------------------------------------------------------------------

def main():
    view = render_sidebar()

    if view == "Customer View":
        render_customer_view()
    else:
        render_company_view()


if __name__ == "__main__":
    main()
