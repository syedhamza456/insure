import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Page title
st.title("📈Insuralyze")
st.markdown("<h2 style='text-align: center; color:rgb(25, 14, 146);'>Analyze Your Insurance Data</h2>", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Check if required columns exist
    required_columns = ["IAP_Industry", "Claim_Amount", "Gross_Premium", "Insurance type", "Co-Insurance Share", "CLIENT", "SumInsured"]
    if all(col in df.columns for col in required_columns):
        # Calculate Loss Ratio on individual level (handle division by zero)
        df["Loss_Ratio"] = np.where(
            df["Gross_Premium"] == 0,
            np.nan,
            df["Claim_Amount"] / df["Gross_Premium"]
        )

        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["Loss_Ratio"], inplace=True)

        # Add a filter for Insurance Type
        insurance_types = df["Insurance type"].unique().tolist()
        insurance_types.append("Both")  # Add option for both types
        selected_insurance_type = st.sidebar.selectbox("Select Insurance Type", options=insurance_types)

        # Filter the DataFrame based on the selected insurance type
        if selected_insurance_type == "Both":
            filtered_df = df
        else:
            filtered_df = df[df["Insurance type"] == selected_insurance_type]

        # --- KPI Calculations ---
        total_claims = filtered_df["Claim_Amount"].sum()
        total_premiums = filtered_df["Gross_Premium"].sum()
        total_sum_insured = filtered_df["SumInsured"].sum()  # Total Sum Insured
        overall_loss_ratio = (total_claims / total_premiums) * 100 if total_premiums > 0 else 0
        overall_premium_rate = (total_premiums / total_sum_insured) * 100 if total_sum_insured > 0 else 0  # Overall Premium Rate

        # Display KPIs in boxes
        st.subheader("Key Performance Indicators (KPIs)")
        kpi_container = st.container()  # Create a container for KPIs
        with kpi_container:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Total Claims (PKR)", value=f"₨ {total_claims:,.2f}")
            with col2:
                st.metric(label="Total Premiums (PKR)", value=f"₨ {total_premiums:,.2f}")
            with col3:
                st.metric(label="Overall Loss Ratio (%)", value=f"{overall_loss_ratio:.2f}%")
            with col4:
                st.metric(label="Overall Premium Rate (%)", value=f"{overall_premium_rate:.2f}%")  # New KPI for Premium Rate

        # --- Industry-wise Loss Ratio and Premium Rate ---
        with st.expander("Industry-Wise Loss Ratio and Premium Rate", expanded=False):
            industry_loss_ratio = filtered_df.groupby("IAP_Industry").agg(
                Total_Claim_Amount=("Claim_Amount", "sum"),
                Total_Gross_Premium=("Gross_Premium", "sum"),
                Total_SumInsured=("SumInsured", "sum")
            ).reset_index()
            industry_loss_ratio["Industry_Loss_Ratio"] = (
                industry_loss_ratio["Total_Claim_Amount"] / industry_loss_ratio["Total_Gross_Premium"]
            ) * 100
            industry_loss_ratio["Premium_Rate"] = (
                industry_loss_ratio["Total_Gross_Premium"] / industry_loss_ratio["Total_SumInsured"]
            ) * 100

            st.write(industry_loss_ratio)

        # --- Client Summary with Industry Aggregation ---
        with st.expander("Client-Wise Summary (with Industry Loss Ratio and Premium Rate)", expanded=False):
            client_summary = filtered_df.groupby(["IAP_Industry", "Insurance type", "CLIENT"]).agg(
                Total_Claim_Amount=("Claim_Amount", "sum"),
                Total_Gross_Premium=("Gross_Premium", "sum"),
                Total_SumInsured=("SumInsured", "sum")
            ).reset_index()
            client_summary["Client_Loss_Ratio"] = (
                client_summary["Total_Claim_Amount"] / client_summary["Total_Gross_Premium"]
            ) * 100
            client_summary["Client_Premium_Rate"] = (
                client_summary["Total_Gross_Premium"] / client_summary["Total_SumInsured"]
            ) * 100

            # Aggregate at the Industry and Insurance Type level for Loss Ratio and Premium Rate
            industry_insurance_summary = filtered_df.groupby(["IAP_Industry", "Insurance type"]).agg(
                Industry_Total_Claim_Amount=("Claim_Amount", "sum"),
                Industry_Total_Gross_Premium=("Gross_Premium", "sum"),
                Industry_Total_SumInsured=("SumInsured", "sum")
            ).reset_index()
            industry_insurance_summary["Industry_Loss_Ratio"] = (
                industry_insurance_summary["Industry_Total_Claim_Amount"] / industry_insurance_summary["Industry_Total_Gross_Premium"]
            ) * 100
            industry_insurance_summary["Industry_Premium_Rate"] = (
                industry_insurance_summary["Industry_Total_Gross_Premium"] / industry_insurance_summary["Industry_Total_SumInsured"]
            ) * 100

            # Merge to include industry loss ratio and premium rate in client summary
            client_summary = pd.merge(client_summary, industry_insurance_summary[["IAP_Industry", "Insurance type", "Industry_Loss_Ratio", "Industry_Premium_Rate"]], on=["IAP_Industry", "Insurance type"], how="left")

            # Create a unique summary for display
            unique_client_summary = client_summary.groupby(["IAP_Industry", "Insurance type"]).agg(
                Clients=("CLIENT", lambda x: ', '.join(x)),
                Total_Claim_Amount=("Total_Claim_Amount", "sum"),
                Total_Gross_Premium=("Total_Gross_Premium", "sum"),
                Total_SumInsured=("Total_SumInsured", "sum"),
                Industry_Loss_Ratio=("Industry_Loss_Ratio", "first"),
                Industry_Premium_Rate=("Industry_Premium_Rate", "first")
            ).reset_index()

            st.write(unique_client_summary)

        # --- ML Model ---
        with st.expander("Machine Learning Predictions", expanded=False):
            X = pd.get_dummies(filtered_df[["IAP_Industry", "Insurance type", "Co-Insurance Share", "CLIENT"]])
            y = filtered_df["Loss_Ratio"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a pipeline with imputation and linear regression
            model = make_pipeline(
                SimpleImputer(strategy='mean'),  # Impute missing values with the mean
                LinearRegression()              # Train a Linear Regression model
            )

            model.fit(X_train, y_train)  # Train the model
            y_pred = model.predict(X_test)  # Make predictions

            mae = mean_absolute_error(y_test, y_pred)
            st.write(f"Mean Absolute Error: {mae:.4f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
            ax.set_xlabel("Actual Loss Ratio")
            ax.set_ylabel("Predicted Loss Ratio")
            ax.set_title("Actual vs. Predicted Loss Ratios")
            st.pyplot(fig)

            predictions_df = pd.DataFrame({
                "IAP_Industry": filtered_df.loc[X_test.index, "IAP_Industry"],
                "Insurance type": filtered_df.loc[X_test.index, "Insurance type"],
                "Co-Insurance Share": filtered_df.loc[X_test.index, "Co-Insurance Share"],
                "CLIENT": filtered_df.loc[X_test.index, "CLIENT"],
                "Actual_Loss_Ratio": y_test,
                "Predicted_Loss_Ratio": y_pred
            })

            industry_summary_predictions = predictions_df.groupby("IAP_Industry").agg(
                Avg_Actual_Loss_Ratio=("Actual_Loss_Ratio", "mean"),
                Avg_Predicted_Loss_Ratio=("Predicted_Loss_Ratio", "mean"),
                Count=("IAP_Industry", "size")
            ).reset_index()

            st.subheader("Industry-Wise Summary of Predictions")
            st.write(industry_summary_predictions)

        # --- Download Results ---
        st.subheader("Download Results")
        output_file = "industry_summary_predictions.xlsx"
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            industry_loss_ratio.to_excel(writer, sheet_name="Industry_Loss_Ratio", index=False)
            unique_client_summary.to_excel(writer, sheet_name="Client_Summary", index=False)
            industry_summary_predictions.to_excel(writer, sheet_name="Industry_Predictions", index=False)

        with open(output_file, "rb") as file:
            st.download_button(
                label="Download All Results",
                data=file,
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error(f"Required columns not found. Please ensure your file contains: {required_columns}")