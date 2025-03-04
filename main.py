import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Custom CSS styling
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        h1 {color: #2b2d42;}
        h2 {color: #2b2d42; border-bottom: 2px solid #ef233c; padding-bottom: 10px;}
        .sidebar .sidebar-content {background-color: #ffffff;}
        .metric-card {background: white; border-radius: 10px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
        .st-bw {background-color: white !important;}
        .css-1aumxhk {background-color: #ffffff; background-image: none; color: #2b2d42}
    </style>
""", unsafe_allow_html=True)

# Page title with icon and better spacing
st.markdown("<h1 style='text-align: center;'><img src='https://cdn-icons-png.flaticon.com/512/906/906334.png' width='60'> Insuralyze</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar with improved styling
with st.sidebar:
    st.header("📤 Data Upload")
    uploaded_file = st.file_uploader("Choose an Excel/CSV file", type=["xlsx", "csv"], help="Ensure your file contains the required columns")
    
    st.markdown("---")
    st.header("🔍 Filters")
    if uploaded_file:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        insurance_types = ["All"] + df["Insurance type"].unique().tolist()
        selected_insurance_type = st.selectbox("Insurance Type", options=insurance_types)
    else:
        selected_insurance_type = "All"

    st.markdown("---")
    st.markdown("### Need Help?")
    st.markdown("Download our [sample template](https://example.com/sample-template.xlsx) to ensure proper data formatting.")

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Check required columns
    required_columns = ["IAP_Industry", "Claim_Amount", "Gross_Premium", "Insurance type", "Co-Insurance Share", "CLIENT", "SumInsured"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"⚠️ Missing required columns. Please ensure your file contains: {', '.join(required_columns)}")
        st.stop()

    # Data processing
    with st.spinner('Crunching numbers...'):
        df["Loss_Ratio"] = np.where(
            df["Gross_Premium"] == 0,
            np.nan,
            df["Claim_Amount"] / df["Gross_Premium"]
        )
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["Loss_Ratio"], inplace=True)

    # Filter data
    filtered_df = df if selected_insurance_type == "All" else df[df["Insurance type"] == selected_insurance_type]

    # --- KPI Section ---
    st.header("📊 Key Metrics")
    
    # First Row (Financial Totals)
    col1, col2 = st.columns(2)
    
    with col1:
        total_claims = filtered_df['Claim_Amount'].sum()
        st.markdown("<div class='metric-card'>💰 **Total Claims (PKR)**<br>" + 
                    f"<h2>{total_claims:,.0f}</h2></div>", 
                    unsafe_allow_html=True)
    
    with col2:
        total_premiums = filtered_df['Gross_Premium'].sum()
        st.markdown("<div class='metric-card'>🏦 **Total Premiums (PKR)**<br>" + 
                    f"<h2>{total_premiums:,.0f}</h2></div>", 
                    unsafe_allow_html=True)

    # Second Row (Performance Ratios)
    col1, col2 = st.columns(2)
    
    with col1:
        overall_loss_ratio = (total_claims / total_premiums * 100) if total_premiums > 0 else 0
        st.markdown("<div class='metric-card'>📉 **Overall Loss Ratio (%)**<br>" + 
                    f"<h2>{overall_loss_ratio:.2f}%</h2></div>", 
                    unsafe_allow_html=True)
    
    with col2:
        overall_premium_rate = (total_premiums / filtered_df['SumInsured'].sum() * 100) if filtered_df['SumInsured'].sum() > 0 else 0
        st.markdown("<div class='metric-card'>📈 **Overall Premium Rate (%)**<br>" + 
                    f"<h2>{overall_premium_rate:.2f}%</h2></div>", 
                    unsafe_allow_html=True)

    # --- Industry Analysis ---
    st.header("🏭 Industry Breakdown")
    with st.expander("View Industry-Wise Metrics", expanded=True):
        industry_stats = filtered_df.groupby("IAP_Industry").agg(
            Total_Claims=('Claim_Amount', 'sum'),
            Total_Premiums=('Gross_Premium', 'sum'),
            Loss_Ratio=('Loss_Ratio', 'mean'),
            Policy_Count=('CLIENT', 'count')
        ).reset_index()
        
        # Format numbers
        industry_stats['Total_Claims'] = industry_stats['Total_Claims'].apply(lambda x: f"{x:,.0f}")
        industry_stats['Total_Premiums'] = industry_stats['Total_Premiums'].apply(lambda x: f"{x:,.0f}")
        industry_stats['Loss_Ratio'] = industry_stats['Loss_Ratio'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            industry_stats.style.highlight_max(subset=['Loss_Ratio'], color='#ffcccc')
            .highlight_min(subset=['Loss_Ratio'], color='#ccffcc'),
            use_container_width=True
        )

    # --- Client Analysis ---
    st.header("🏢 Client Overview")
    tab1, tab2 = st.tabs(["Summary View", "Detailed Analysis"])
    
    with tab1:
        client_summary = filtered_df.groupby("CLIENT").agg(
            Policies=('CLIENT', 'count'),
            Total_Exposure=('SumInsured', 'sum'),
            Avg_Loss_Ratio=('Loss_Ratio', 'mean')
        ).reset_index()
        st.dataframe(client_summary.style.bar(subset=['Avg_Loss_Ratio'], color='#ef233c55'), use_container_width=True)
    
    with tab2:
        client_details = filtered_df.groupby(["CLIENT", "Insurance type"]).agg({
            'Claim_Amount': 'sum',
            'Gross_Premium': 'sum',
            'SumInsured': 'sum'
        }).reset_index()
        st.dataframe(client_details, use_container_width=True)

    # --- Machine Learning Section ---
    st.header("🤖 Predictive Analysis")
    with st.expander("View Prediction Model Details"):
        # Model training
        X = pd.get_dummies(filtered_df[["IAP_Industry", "Insurance type", "Co-Insurance Share", "CLIENT"]])
        y = filtered_df["Loss_Ratio"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = make_pipeline(
            SimpleImputer(strategy='mean'),
            LinearRegression()
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, c='#2b2d42', alpha=0.6, edgecolors='w')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', lw=2, color='#ef233c')
        ax.set_xlabel('Actual Loss Ratio', fontsize=12)
        ax.set_ylabel('Predicted Loss Ratio', fontsize=12)
        ax.set_title('Model Performance', pad=20)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        st.metric("Model Accuracy (MAE)", f"{mae:.4f}")

    # --- Download Section ---
    st.markdown("---")
    st.header("📥 Export Results")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download Industry Report",
                data=industry_stats.to_csv(index=False).encode('utf-8'),
                file_name='industry_report.csv',
                mime='text/csv'
            )
        with col2:
            st.download_button(
                label="💾 Full Data Export",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name='full_analysis.csv',
                mime='text/csv'
            )

else:
    st.info("👋 Welcome to Insuralyze! Please upload your insurance data to begin analysis.")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=200)
    st.markdown("""
        ### Getting Started:
        1. Upload your insurance data file (Excel/CSV)
        2. Apply filters using the sidebar
        3. Explore different analysis sections
        4. Download reports as needed
    """)
