import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page configuration and wide layout
st.set_page_config(page_title="Revenue Operations Dashboard", layout="wide")

##########################################
# 1. Load Data
##########################################
@st.cache_data
def load_sales_data():
    df = pd.read_csv("revops_data.csv")
    df["Deal_Size (£)"] = df["Deal_Size (£)"].round(2)
    return df

@st.cache_data
def load_customer_success_data():
    customer_success_df = pd.read_csv("customer_success_data.csv")
    return customer_success_df

df = load_sales_data()
customer_success_df = load_customer_success_data()

##########################################
# 2. Pre-Calculations for Revenue Headline Metrics
##########################################

# Define New Customers, Renewals, and Churned Customers
new_customers = df[(df["Deal_Stage"] == "Closed Won") & (df["Opportunity_Type"] == "New Customer")]
renewals = df[(df["Deal_Stage"] == "Closed Won") & (df["Opportunity_Type"] == "Renewal")]
churned_customers = customer_success_df[customer_success_df["Renewal_Status"] == "Churned"]

total_customers = new_customers.shape[0] + renewals.shape[0] - churned_customers.shape[0]

# Total Revenue Calculation
total_revenue = df[df["Deal_Stage"] == "Closed Won"]["Deal_Size (£)"].sum()

# Net Revenue Retention (NRR) %
nrr = ((renewals["Expansion_Revenue (£)"].sum() + renewals["Deal_Size (£)"].sum()) / renewals["Deal_Size (£)"].sum()) * 100

# Gross Revenue Retention (GRR) %
grr = (renewals["Deal_Size (£)"].sum() / (renewals["Deal_Size (£)"].sum() + churned_customers["Churned_Revenue (£)"].sum())) * 100

# Churn Rate %
churn_rate = (churned_customers.shape[0] / (renewals.shape[0] + churned_customers.shape[0])) * 100

# Expansion Revenue %
expansion_revenue_pct = (renewals["Expansion_Revenue (£)"].sum() / total_revenue) * 100

##########################################
# 3. Revenue Headline Metrics
##########################################
st.title("Revenue Operations Dashboard")

with st.expander("Revenue Headline Metrics", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue (ARR)", f"£{total_revenue:,.2f}")
    col2.metric("Net Revenue Retention (NRR)", f"{nrr:.2f}%")
    col3.metric("Gross Revenue Retention (GRR)", f"{grr:.2f}%")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")
    col5.metric("Expansion Revenue %", f"{expansion_revenue_pct:.2f}%")

##########################################
# 3. Revenue Headline Metrics
##########################################
st.title("Revenue Operations Dashboard")

with st.expander("Revenue Headline Metrics", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue (ARR)", f"£{total_revenue:,.2f}")
    col2.metric("Net Revenue Retention (NRR)", f"{nrr:.2f}%")
    col3.metric("Gross Revenue Retention (GRR)", f"{grr:.2f}%")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")
    col5.metric("Expansion Revenue %", f"{expansion_revenue_pct:.2f}%")
# --------------------------------------------------
# Revenue Overview: MRR/ARR trends & revenue breakdown
# --------------------------------------------------
with st.expander("Revenue Overview"):
    # Aggregate revenue from closed won deals by Region and Segment
    closed_won_rev = (
        df[df["Deal_Stage"] == "Closed Won"]
        .groupby(["Region", "Segment"])
        .agg(Total_Revenue=("Deal_Size (£)", "sum"))
        .reset_index()
    )
    
    # Create a Plotly Express bar chart
    fig = px.bar(
        closed_won_rev,
        x="Region",
        y="Total_Revenue",
        color="Segment",          # Use Segment as the hue/color
        barmode="group",          # "group" mode shows side-by-side bars; try "stack" if preferred
        title="Total Closed Won Revenue by Region & Segment",
        labels={"Total_Revenue": "Total Revenue (£)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # If your data contains a closed date and ARR/MRR column, you can show a trend.
    if "Closed_Date" in df.columns:
        df_trend = df[df["Deal_Stage"] == "Closed Won"].dropna(subset=["Closed_Date"]).copy()
        df_trend["Closed_Date"] = pd.to_datetime(df_trend["Closed_Date"], errors="coerce")
        df_trend["Month"] = df_trend["Closed_Date"].dt.to_period("M").dt.to_timestamp()
        revenue_trend = df_trend.groupby("Month").agg(Total_ARR=("Deal_Size (£)", "sum")).reset_index()
        fig_trend = px.line(
            revenue_trend, x="Month", y="Total_ARR", markers=True,
            title="ARR Trend Over Time",
            labels={"Total_ARR": "Total ARR (£)", "Month": "Month"}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("ARR/MRR trend data not available.")

   # --------------------------------------------------
# Sales Performance: Pipeline analysis, win rates, sales cycle
# --------------------------------------------------
with st.expander("Sales Performance"):

    # Ensure Sales Cycle Days column is numeric
    df["Sales_Cycle_Days"] = pd.to_numeric(df["Sales_Cycle_Days"], errors="coerce")

    # Create a Plotly Box Plot
    fig_sales_cycle = px.box(
        df[df["Deal_Stage"] == "Closed Won"], 
        x="Region",
        y="Sales_Cycle_Days",
        color="Region",
        title="Distribution of Sales Cycle Length by Region",
        labels={"Sales_Cycle_Days": "Sales Cycle Duration (Days)", "Region": "Region"},
        points="all"  # Show all data points (outliers included)
    )

    # Improve Layout & Readability
    fig_sales_cycle.update_layout(
        width=1000,  # Adjust width for clarity
        height=500,  # Adjust height
        xaxis_title="Region",
        yaxis_title="Sales Cycle Duration (Days)",
        boxmode="group"
    )

    # Display in Streamlit
    st.plotly_chart(fig_sales_cycle, use_container_width=True, key="sales_cycle")
    

# Filter out Closed Won and Closed Lost
    filtered_df = df[~df["Deal_Stage"].isin(["Closed Won", "Closed Lost"])]

    stacked_bar_data = (
        filtered_df.groupby(["Region", "Segment", "Deal_Stage"])
        .agg({"Opportunity_ID": "count"})
        .reset_index()
        .rename(columns={"Opportunity_ID": "Deal Count"})
)

    fig_stacked = px.bar(
        stacked_bar_data,
        x="Deal_Stage",
        y="Deal Count",
        color="Segment",
        barmode="stack",
        facet_col="Region",
        title="Pipeline Breakdown by Region & Segment",
        labels={"Deal_Stage": "Pipeline Stage", "Deal Count": "Total Deals", "Segment": "Customer Segment"}
)

# Remove the "Region=" prefix from facet annotations
    for annotation in fig_stacked.layout.annotations:
        annotation.text = annotation.text.split("=")[-1]

    st.plotly_chart(fig_stacked, use_container_width=True, key="pipeline_breakdown")

    total_deals = df.groupby(["Region", "Segment"])["Opportunity_ID"].count()
    closed_won  = df[df["Deal_Stage"] == "Closed Won"].groupby(["Region", "Segment"])["Opportunity_ID"].count()
    win_rate_df = (closed_won / total_deals * 100).fillna(0).reset_index()
    win_rate_df.columns = ["Region", "Segment", "Win Rate (%)"]
    win_rate_df["Win Rate (%)"] = win_rate_df["Win Rate (%)"].round(2)
    heatmap_data = win_rate_df.pivot(index="Segment", columns="Region", values="Win Rate (%)").astype(float)
    fig_heatmap = px.imshow(
        heatmap_data,
        color_continuous_scale="Blues",
        title="Win Rate Heatmap by Region & Segment",
        labels={"color": "Win Rate (%)"}
    )
    fig_heatmap.update_traces(
        text=heatmap_data.applymap(lambda x: f"{x:.2f}"),
        texttemplate="%{text}"
    )
    fig_heatmap.update_layout(
        width=1000,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_title="",
        yaxis_title="",
        coloraxis_colorbar=dict(title="Win Rate (%)")
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
# --------------------------------------------------
# Marketing Performance: Campaign performance, MQL/SQL conversion
# --------------------------------------------------
with st.expander("Marketing Performance"):
    campaign_type = lead_df[lead_df["Lead_Source"] == "Marketing Campaign"].copy()
    campaign_type["Lead_Creation_Date"] = pd.to_datetime(campaign_type["Lead_Creation_Date"])
    campaign_type["Lead_Creation_Month"] = campaign_type["Lead_Creation_Date"].dt.to_period("M").astype(str)
    monthly_trend = campaign_type.groupby(["Lead_Creation_Month", "Campaign_Type"]).agg(
        Total_Leads=("Lead_ID", "count")
    ).reset_index()
    fig_monthly = px.line(
        monthly_trend,
        x="Lead_Creation_Month",
        y="Total_Leads",
        color="Campaign_Type",
        title="Monthly Lead Volume by Campaign Type",
        labels={"Total_Leads": "Total Leads", "Lead_Creation_Month": "Month"}
    )
    st.plotly_chart(fig_monthly, use_container_width=True, key="monthly_leads")

    # Flag conversion if not already defined
    if "Converted_Flag" not in campaign_type.columns:
        campaign_type["Converted_Flag"] = np.where(campaign_type["Converted_to_Opportunity"] == "Yes", 1, 0)
    campaign_type["Opportunity_Creation_Month"] = campaign_type.apply(
        lambda row: row["Lead_Creation_Month"] if row["Converted_to_Opportunity"] == "Yes" else np.nan,
        axis=1
    )
    opp_trend = campaign_type.dropna(subset=["Opportunity_Creation_Month"]).groupby("Opportunity_Creation_Month").agg(
        Opportunities_Generated=("Lead_ID", "count")
    ).reset_index()
    opp_trend["Opportunity_Creation_Month"] = opp_trend["Opportunity_Creation_Month"].astype(str)
    fig_opp = px.line(
        opp_trend,
        x="Opportunity_Creation_Month",
        y="Opportunities_Generated",
        title="Opportunities Generated Over Time",
        labels={"Opportunities_Generated": "Opportunities Generated", "Opportunity_Creation_Month": "Month"}
    )
    st.plotly_chart(fig_opp, use_container_width=True, key="opportunities_generated")

    conv_seg = campaign_type.groupby(["Segment", "Campaign_Type"]).agg(
        Total_Leads=("Lead_ID", "count"),
        Converted_Leads=("Converted_Flag", "sum")
    ).reset_index()
    conv_seg["Conversion_Rate"] = (conv_seg["Converted_Leads"] / conv_seg["Total_Leads"]) * 100
    fig_conv = px.bar(
        conv_seg,
        x="Segment",
        y="Conversion_Rate",
        color="Campaign_Type",
        barmode="group",
        title="Conversion Rate by Campaign Type & Segment",
        labels={"Conversion_Rate": "Conversion Rate (%)", "Segment": "Segment", "Campaign_Type": "Campaign Type"}
    )
    st.plotly_chart(fig_conv, use_container_width=True, key="conversion_rate")

    lead_seg = lead_df.groupby(["Segment", "Lead_Source"]).agg(
        Total_Leads=("Lead_ID", "count")
    ).reset_index()
    fig_lead = px.bar(
        lead_seg,
        x="Segment",
        y="Total_Leads",
        color="Lead_Source",
        barmode="stack",
        title="Lead Distribution by Segment & Source",
        labels={"Total_Leads": "Total Leads", "Segment": "Segment", "Lead_Source": "Lead Source"}
    )
    st.plotly_chart(fig_lead, use_container_width=True, key="lead_distribution")

    conv_region = campaign_type.groupby(["Region", "Segment"]).agg(
        Total_Leads=("Lead_ID", "count"),
        Converted_Leads=("Converted_Flag", "sum")
    ).reset_index()
    conv_region["Conversion_Rate"] = (conv_region["Converted_Leads"] / conv_region["Total_Leads"]) * 100
    heatmap_conv = conv_region.pivot(index="Region", columns="Segment", values="Conversion_Rate")
    fig_heat_conv = px.imshow(
        heatmap_conv,
        labels={"color": "Conversion Rate (%)"},
        title="Lead Conversion Rate by Region & Segment",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_heat_conv, use_container_width=True, key="conversion_heatmap")

# --------------------------------------------------
# Customer Success: NPS scores, expansion revenue (placeholder)
# --------------------------------------------------
with st.expander("Customer Success"):
    customer_counts = (
        new_customers.groupby("Region")["Opportunity_ID"].count() +
        renewals.groupby("Region")["Opportunity_ID"].count() -
        churned_customers.groupby("Region")["Opportunity_ID"].count()
    ).reset_index(name="Total_Customers")
    fig_customers = px.bar(
        customer_counts,
        x="Region",
        y="Total_Customers",
        title="Customer Counts by Region",
        labels={"Total_Customers": "Total Customers"}
    )
    st.plotly_chart(fig_customers, use_container_width=True,key="customer_counts")
    fig_nps = px.histogram(
        customer_success_df, x="NPS_Score", color="Region", nbins=20, 
        title="NPS Score Distribution by Region", labels={"NPS_Score": "NPS Score"}, 
        barmode="overlay"
    )
    st.plotly_chart(fig_nps, use_container_width=True, key="nps_distribution")

    expansion_rev = customer_success_df.groupby("Segment").agg(
        Total_Expansion_Revenue=("Expansion_Revenue (£)", "sum")
    ).reset_index()
    fig_expansion = px.bar(
        expansion_rev, x="Segment", y="Total_Expansion_Revenue", 
        title="Total Expansion Revenue by Segment", 
        labels={"Total_Expansion_Revenue": "Total Expansion Revenue (£)"},
        color="Segment"
    )
    st.plotly_chart(fig_expansion, use_container_width=True, key="expansion_revenue")

    churn_counts = customer_success_df.groupby(["Region", "Renewal_Status"]).size().reset_index(name="Count")
    fig_churn = px.bar(
        churn_counts, x="Region", y="Count", color="Renewal_Status", 
        title="Customer Churn vs Retention by Region"
    )
    st.plotly_chart(fig_churn, use_container_width=True, key="customer_churn")


st.markdown("### End of Dashboard")
