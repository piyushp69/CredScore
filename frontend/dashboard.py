import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import pickle

# Setup layout
st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦", layout="wide")

# Custom CSS for glassmorphism and modern feel
st.markdown("""
<style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sleek card-like containers for form elements */
    div.stForm {
        background: rgba(30, 30, 30, 0.6);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Subheaders and markdown */
    p, span, div.stMarkdown {
        color: #b0b0b0;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00d2ff;
    }
</style>
""", unsafe_allow_html=True)

# Load model for global insights
@st.cache_resource
def load_model():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    with open(os.path.join(models_dir, 'xgboost_model.pkl'), 'rb') as f:
        clf = pickle.load(f)
    with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    return clf, feature_names

try:
    clf, feature_names = load_model()
except Exception as e:
    st.error("Could not load local models for global insights. Make sure they are trained.")
    feature_names = []

# Sidebar Navigation
with st.sidebar:
    st.title("🏦 Credit Risk ML")
    st.markdown("---")
    nav = st.radio("Navigation", ["🔍 Applicant Underwriting", "📊 Portfolio Insights"])
    st.markdown("---")
    st.info("💡 **Tip**: Use normalized values for external sources (0.0 to 1.0).")

# Main Content
if nav == "🔍 Applicant Underwriting":
    st.title("Applicant Underwriting")
    st.markdown("Evaluate new loan applicants instantly with our ML model.")
    
    with st.form("applicant_form"):
        st.subheader("Applicant Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            ext_source_1 = st.number_input("External Source 1 (Norm)", value=0.50, min_value=0.0, max_value=1.0, step=0.05)
            ext_source_2 = st.number_input("External Source 2 (Norm)", value=0.50, min_value=0.0, max_value=1.0, step=0.05)
        with col2:
            ext_source_3 = st.number_input("External Source 3 (Norm)", value=0.50, min_value=0.0, max_value=1.0, step=0.05)
            days_birth = st.number_input("Age (in days, negative)", value=-15000, max_value=0, step=100)
        with col3:
            amt_credit = st.number_input("Credit Amount ($)", value=500000.0, min_value=0.0, step=10000.0)
            amt_annuity = st.number_input("Annuity Amount ($)", value=25000.0, min_value=0.0, step=1000.0)
            
        submit = st.form_submit_button("Analyze Applicant & Generate Decision", use_container_width=True)
        
    if submit:
        payload = {
            "features": {
                "EXT_SOURCE_1": ext_source_1,
                "EXT_SOURCE_2": ext_source_2,
                "EXT_SOURCE_3": ext_source_3,
                "DAYS_BIRTH": days_birth,
                "AMT_CREDIT": amt_credit,
                "AMT_ANNUITY": amt_annuity
            }
        }
        
        with st.spinner("Analyzing risk profile..."):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                response.raise_for_status()
                result = response.json()
                
                risk_score = result["probability_of_default"]
                pos_contrib = result["top_3_positive_contributions"]
                neg_contrib = result["top_3_negative_contributions"]
                
                st.success("✅ Analysis Complete")
                if risk_score < 0.3:
                    st.balloons()
                
                st.markdown("---")
                st.subheader("Underwriting Decision")
                
                c1, c2, c3 = st.columns([1, 1, 1.2])
                
                with c1:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score * 100,
                        number = {"suffix": "%", "font": {"color": "#ffffff"}},
                        title = {'text': "Default Probability", 'font': {"color": "#b0b0b0", "size": 18}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                            'bar': {'color': "rgba(255, 255, 255, 0.4)"},
                            'bgcolor': "#1e1e1e",
                            'steps': [
                                {'range': [0, 30], 'color': "#00ff88"},
                                {'range': [30, 70], 'color': "#ffaa00"},
                                {'range': [70, 100], 'color': "#ff3333"}
                            ],
                            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': risk_score * 100}
                        }
                    ))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#ffffff"}, margin=dict(t=50, b=10, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)
                    
                with c2:
                    st.markdown("**Profile Breakdown**")
                    # Radar Chart for applicant inputs
                    categories = ['EXT 1', 'EXT 2', 'EXT 3', 'Age (Normalized)', 'Credit (Norm)', 'Annuity (Norm)']
                    # Normalize age and credit just for radar visual
                    norm_age = max(0, min(1, (-days_birth - 10000) / 15000))
                    norm_credit = max(0, min(1, amt_credit / 2000000))
                    norm_annuity = max(0, min(1, amt_annuity / 100000))
                    values = [ext_source_1, ext_source_2, ext_source_3, norm_age, norm_credit, norm_annuity]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        fillcolor='rgba(0, 210, 255, 0.3)',
                        line=dict(color='#00d2ff', width=2)
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)"),
                            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)")
                        ),
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#b0b0b0"),
                        margin=dict(t=30, b=30, l=30, r=30)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with c3:
                    st.markdown("**Why was this score assigned? (SHAP)**")
                    features = list(pos_contrib.keys()) + list(neg_contrib.keys())
                    values = list(pos_contrib.values()) + list(neg_contrib.values())
                    colors = ['#ff3333' if v > 0 else '#00ff88' for v in values]
                    
                    fig2 = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker_color=colors,
                        marker_line_color='rgba(255,255,255,0.2)',
                        marker_line_width=1
                    ))
                    fig2.update_layout(
                        xaxis_title="Risk Impact (SHAP)",
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#b0b0b0"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                        margin=dict(t=10, b=40, l=10, r=10)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error connecting to backend API: {e}")

else:
    st.title("Global Portfolio Insights")
    st.markdown("Interactive visualizations exploring overall dataset trends and model feature importances.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Global Feature Importances")
        if feature_names:
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig3 = px.bar(imp_df, x='Importance', y='Feature', orientation='h', 
                          color='Importance', color_continuous_scale=['#1e1e1e', '#00d2ff'])
            fig3.update_layout(
                yaxis={'categoryorder':'total ascending'},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#b0b0b0"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Model not found. Train the model to see feature importances.")
            
    with col2:
        st.subheader("Risk Distribution by Income Bracket")
        # Enhancing the dummy data visual to look premium
        income_brackets = ["0-50k", "50k-100k", "100k-150k", "150k+"]
        avg_risk = [15.2, 8.5, 4.1, 2.3]
        
        fig4 = px.area(x=income_brackets, y=avg_risk, labels={'x': 'Income Bracket', 'y': 'Default Rate (%)'},
                      markers=True)
        
        fig4.update_traces(line_color="#ff3333", fillcolor="rgba(255, 51, 51, 0.2)", marker=dict(size=10, color="#ffffff"))
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b0b0b0"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[0, 20]),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig4, use_container_width=True)
