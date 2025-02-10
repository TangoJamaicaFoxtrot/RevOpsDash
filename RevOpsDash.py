import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Revenue Operations Dashboard", layout="wide")

##########################################
# Load Data from SQLite Database
##########################################
@st.cache_data
def load_data():
    conn = sqlite3.connect("revops_database.db")

    opportunity_df = pd.read_sql("SELECT * FROM opportunity_data", conn)
    lead_df = pd.read_sql("SELECT * FROM lead_data", conn)
    customer_success_df = pd.read_sql("SELECT * FROM customer_success", conn)

    conn.close()
    return opportunity_df, lead_df, customer_success_df

# Load datasets
opportunity_df, lead_df, customer_success_df = load_data()

# Convert dates
opportunity_df["Created_Date"] = pd.to_datetime(opportunity_df["Created_Date"], errors="coerce")
opportunity_df["Closed_Date"] = pd.to_datetime(opportunity_df["Closed_Date"], errors="coerce")

##########################################
# Headline Metrics
##########################################
st.title("Revenue Operations Dashboard")

with st.expander("Headline Metrics", expanded=True):
    total_customers = opportunity_df["Customer_ID"].nunique()
    new_customers = opportunity_df[opportunity_df["Opportunity_Type"] == "New Customer"]["Customer_ID"].nunique()
    renewals = opportunity_df[opportunity_df["Opportunity_Type"] == "Renewal"]["Customer_ID"].nunique()
    churned_customers = opportunity_df[(opportunity_df["Opportunity_Type"] == "Renewal") & (opportunity_df["Deal_Stage"] == "Closed Lost")]["Customer_ID"].nunique()

    win_loss_counts = opportunity_df["Deal_Stage"].value_counts()
    overall_win_rate = (win_loss_counts.get("Closed Won", 0) /
                        (win_loss_counts.get("Closed Won", 0) + win_loss_counts.get("Closed Lost", 0))) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers", f"{total_customers}")
    col2.metric("New Customers", f"{new_customers}")
    col3.metric("Renewals", f"{renewals}")
    col4.metric("Churned Customers", f"{churned_customers}")
    col5.metric("Overall Win Rate", f"{overall_win_rate:.2f}%")

##########################################
# Revenue Overview
##########################################
with st.expander("Revenue Overview"):
    st.markdown("**Closed Won Revenue by Region & Segment**")

    closed_won_rev = (
        opportunity_df[opportunity_df["Deal_Stage"] == "Closed Won"]
        .groupby(["Region", "Segment"])
        .agg(Total_Revenue=("Deal_Size", "sum"))
        .reset_index()
    )

    fig_rev = px.bar(
        closed_won_rev,
        x="Region",
        y="Total_Revenue",
        color="Segment",
        barmode="group",
        title="Total Closed Won Revenue by Region & Segment",
        labels={"Total_Revenue": "Total Revenue (£)"}
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("**ARR Trend Over Time**")
    if "Closed_Date" in opportunity_df.columns:
        revenue_trend = (
            opportunity_df[opportunity_df["Deal_Stage"] == "Closed Won"]
            .dropna(subset=["Closed_Date"])
            .groupby(opportunity_df["Closed_Date"].dt.to_period("M"))
            .agg(Total_ARR=("Deal_Size", "sum"))
            .reset_index()
        )
        revenue_trend["Closed_Date"] = revenue_trend["Closed_Date"].astype(str)

        fig_trend = px.line(
            revenue_trend, x="Closed_Date", y="Total_ARR", markers=True,
            title="ARR Trend Over Time",
            labels={"Total_ARR": "Total ARR (£)", "Closed_Date": "Month"}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("ARR/MRR trend data not available.")

##########################################
# Sales Performance
##########################################
with st.expander("Sales Performance"):
    st.markdown("**Sales Cycle Distribution by Region**")

    opportunity_df["Sales_Cycle_Days"] = pd.to_numeric(opportunity_df["Closed_Date"] - opportunity_df["Created_Date"]).dt.days

    fig_sales_cycle = px.box(
        opportunity_df[opportunity_df["Deal_Stage"] == "Closed Won"],
        x="Region",
        y="Sales_Cycle_Days",
        color="Region",
        title="Distribution of Sales Cycle by Region",
        labels={"Sales_Cycle_Days": "Sales Cycle Duration (Days)", "Region": "Region"},
        points="all"
    )
    st.plotly_chart(fig_sales_cycle, use_container_width=True)

##########################################
# Customer Acquisition & Retention
##########################################
with st.expander("Customer Acquisition & Retention"):
    st.markdown("**Customer Counts by Region**")

    customer_counts = (
        opportunity_df.groupby("Region")["Customer_ID"].nunique()
        .reset_index(name="Total_Customers")
    )

    fig_customers = px.bar(
        customer_counts,
        x="Region",
        y="Total_Customers",
        title="Customer Counts by Region",
        labels={"Total_Customers": "Total Customers"}
    )
    st.plotly_chart(fig_customers, use_container_width=True)

##########################################
# Marketing Performance
##########################################
with st.expander("Marketing Performance"):
    st.markdown("**Monthly Lead Volume by Campaign Type**")

    lead_df["Lead_Creation_Date"] = pd.to_datetime(lead_df["Lead_Creation_Date"], errors="coerce")
    lead_df["Lead_Creation_Month"] = lead_df["Lead_Creation_Date"].dt.to_period("M").astype(str)

    monthly_trend = lead_df.groupby(["Lead_Creation_Month", "Campaign_Type"]).agg(Total_Leads=("Lead_ID", "count")).reset_index()

    fig_lead_trend = px.line(
        monthly_trend,
        x="Lead_Creation_Month",
        y="Total_Leads",
        color="Campaign_Type",
        title="Monthly Lead Volume by Campaign Type",
        labels={"Total_Leads": "Total Leads", "Lead_Creation_Month": "Month"}
    )
    st.plotly_chart(fig_lead_trend, use_container_width=True)

##########################################
# Customer Success
##########################################
with st.expander("Customer Success"):
    st.markdown("**Customer Success Overview**")

    customer_success_df["Renewal_Status"] = customer_success_df["Renewal_Status"].astype(str)

    renewal_status_counts = customer_success_df["Renewal_Status"].value_counts().reset_index()
    renewal_status_counts.columns = ["Renewal_Status", "Count"]

    fig_renewal_status = px.pie(
        renewal_status_counts,
        names="Renewal_Status",
        values="Count",
        title="Customer Renewal Status Distribution"
    )
    st.plotly_chart(fig_renewal_status, use_container_width=True)

    st.markdown("**CSAT Scores Distribution**")

    fig_csat = px.histogram(
        customer_success_df,
        x="CSAT_Score",
        nbins=10,
        title="CSAT Score Distribution",
        labels={"CSAT_Score": "CSAT Score"}
    )
    st.plotly_chart(fig_csat, use_container_width=True)

st.markdown("### End of Dashboard")
