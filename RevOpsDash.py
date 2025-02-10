import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Set wide layout and a page title
st.set_page_config(page_title="Revenue Operations Dashboard", layout="wide")

##########################################
# 1. Load Data (with caching for performance)
##########################################
@st.cache_data
def load_sales_data():
    # Update the file path as needed
    df = pd.read_csv("/Users/anastasia/Downloads/revops_data.csv")
    df["Region"] = df["Region"].fillna("NA")
    df["Deal_Size (£)"] = df["Deal_Size (£)"].round(2)
    return df

@st.cache_data
def load_lead_data():
    # Update the file path as needed
    lead_df = pd.read_csv("/Users/anastasia/Downloads/updated_lead_data_with_strategic.csv")
    lead_df["Region"] = lead_df["Region"].fillna("NA")
    return lead_df

df = load_sales_data()
lead_df = load_lead_data()

##########################################
# 2. Title and Data Overview
##########################################
st.title("Revenue Operations Dashboard")

st.markdown("### Sales Data Overview")
st.write("Below are the first 5 rows of the sales data:")
st.dataframe(df.head())

##########################################
# 3. Customer Metrics
##########################################
st.markdown("## Customer Metrics")

# Filter data by opportunity type and deal stage
new_customers = df[(df["Deal_Stage"] == "Closed Won") & (df["Opportunity_Type"] == "New Customer")]
renewals = df[(df["Deal_Stage"] == "Closed Won") & (df["Opportunity_Type"] == "Renewal")]
churned_customers = df[(df["Deal_Stage"] == "Closed Lost") & (df["Opportunity_Type"] == "Renewal")]

# Calculate total active customers (new + renewals – churned)
total_customers = new_customers.shape[0] + renewals.shape[0] - churned_customers.shape[0]

st.write(f"**Total Customers:** {total_customers}")
st.write(f"**New Customers:** {new_customers.shape[0]}")
st.write(f"**Renewals:** {renewals.shape[0]}")
st.write(f"**Churned Customers:** {churned_customers.shape[0]}")

##########################################
# 4. Win Rate Analysis
##########################################
st.markdown("## Win Rate Analysis")

# Overall win rate calculation
win_loss_counts = df["Deal_Stage"].value_counts()
overall_win_rate = (win_loss_counts.get("Closed Won", 0) / 
                    (win_loss_counts.get("Closed Won", 0) + win_loss_counts.get("Closed Lost", 0)) * 100)
st.write(f"**Overall Win Rate:** {overall_win_rate:.2f}%")

# Win rate by Region
total_deals_by_region = df.groupby("Region")["Opportunity_ID"].count()
closed_won_by_region = df[df["Deal_Stage"] == "Closed Won"].groupby("Region")["Opportunity_ID"].count()
win_rate_by_region = (closed_won_by_region / total_deals_by_region * 100).fillna(0)
st.write("**Win Rate by Region:**")
st.dataframe(win_rate_by_region.reset_index())

# Win rate by Segment
total_deals_by_segment = df.groupby("Segment")["Opportunity_ID"].count()
closed_won_by_segment = df[df["Deal_Stage"] == "Closed Won"].groupby("Segment")["Opportunity_ID"].count()
win_rate_by_segment = (closed_won_by_segment / total_deals_by_segment * 100).fillna(0)
st.write("**Win Rate by Segment:**")
st.dataframe(win_rate_by_segment.reset_index())

##########################################
# 5. Revenue Breakdown & Closed Lost Analysis
##########################################
st.markdown("## Revenue Breakdown & Closed Lost Analysis")

with st.expander("Closed Won Revenue Breakdown"):
    closed_won_df = (
        df[df["Deal_Stage"] == "Closed Won"]
        .groupby(["Region", "Segment", "Opportunity_Type"])
        .agg(
            Total_Revenue=("Deal_Size (£)", "sum"), 
            Closed_Won_Deals=("Opportunity_ID", "count")
        )
        .reset_index()
    )
    st.dataframe(closed_won_df.head())

with st.expander("Closed Lost Analysis"):
    closed_lost_df = (
        df[df["Deal_Stage"] == "Closed Lost"]
        .groupby(["Region", "Segment", "Opportunity_Type"])
        .agg(
            Closed_Lost_Count=("Opportunity_ID", "count"),
            Closed_Lost_Value=("Deal_Size (£)", "sum")
        )
        .reset_index()
    )
    st.dataframe(closed_lost_df.head())
    
    closed_lost_reasons = df[df["Deal_Stage"] == "Closed Lost"]["Closed_Lost_Reason"].value_counts()
    st.write("**Closed Lost Reasons:**")
    st.dataframe(closed_lost_reasons.reset_index().rename(columns={"index": "Reason", "Closed_Lost_Reason": "Count"}))

##########################################
# 6. Pipeline Analysis
##########################################
st.markdown("## Pipeline Analysis")

with st.expander("Pipeline Summary"):
    pipe_summary = (
        df.groupby(["Region", "Deal_Stage"])
        .agg(Deal_Count=("Opportunity_ID", "count"), Deal_Value=("Deal_Size (£)", "sum"))
        .reset_index()
    )
    st.dataframe(pipe_summary.head(6))

with st.expander("Sales Cycle Analysis (Closed Won Deals)"):
    sales_cycle_analysis = (
        df[df["Deal_Stage"] == "Closed Won"]
        .groupby(["Region", "Segment"])
        .agg(Avg_Sales_Cycle_Days=("Sales_Cycle_Days", "mean"),
             Median_Sales_Cycle_Days=("Sales_Cycle_Days", "median"))
        .reset_index()
    )
    st.dataframe(sales_cycle_analysis)

with st.expander("Salesperson Leaderboards"):
    salesperson_leaderboards = (
        df.groupby(["Salesperson", "Deal_Stage"])
        .agg(Stage_Value=("Deal_Size (£)", "sum"),
             Stage_Count=("Opportunity_ID", "count"))
        .reset_index()
    )
    st.dataframe(salesperson_leaderboards.head(15))

##########################################
# 7. Win Rate Heatmap by Region & Segment (Plotly)
##########################################
st.markdown("## Win Rate Heatmap by Region & Segment")

# Compute win rates per Region & Segment
total_deals = df.groupby(["Region", "Segment"])["Opportunity_ID"].count()
closed_won = df[df["Deal_Stage"] == "Closed Won"].groupby(["Region", "Segment"])["Opportunity_ID"].count()
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
# Add text annotations manually
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

##########################################
# 8. Win Rate Bar Chart by Region & Segment (Plotly)
##########################################
st.markdown("## Win Rate Bar Chart by Region & Segment")

# Prepare data for bar chart
total_deals_by_region = df.groupby("Region")["Opportunity_ID"].count()
closed_won_by_region = df[df["Deal_Stage"] == "Closed Won"].groupby("Region")["Opportunity_ID"].count()
win_rate_by_region_df = (closed_won_by_region / total_deals_by_region * 100).fillna(0).reset_index()
win_rate_by_region_df.columns = ["Category", "Win Rate (%)"]
win_rate_by_region_df["Type"] = "Region"

total_deals_by_segment = df.groupby("Segment")["Opportunity_ID"].count()
closed_won_by_segment = df[df["Deal_Stage"] == "Closed Won"].groupby("Segment")["Opportunity_ID"].count()
win_rate_by_segment_df = (closed_won_by_segment / total_deals_by_segment * 100).fillna(0).reset_index()
win_rate_by_segment_df.columns = ["Category", "Win Rate (%)"]
win_rate_by_segment_df["Type"] = "Segment"

win_rate_data = pd.concat([win_rate_by_region_df, win_rate_by_segment_df])
max_win_rate = win_rate_data["Win Rate (%)"].max()
y_axis_limit = min(100, max_win_rate + 10)

fig_bar = px.bar(
    win_rate_data,
    x="Category",
    y="Win Rate (%)",
    color="Type",
    barmode="group",
    title="Win Rate by Region & Segment",
    labels={"Category": "Region / Segment", "Win Rate (%)": "Win Rate (%)"},
    color_discrete_map={"Region": "blue", "Segment": "green"}
)
fig_bar.update_layout(yaxis=dict(range=[0, y_axis_limit]))
st.plotly_chart(fig_bar, use_container_width=True)

##########################################
# 9. Closed Lost Reasons (Seaborn)
##########################################
st.markdown("## Closed Lost Reasons (Seaborn)")
plt.figure(figsize=(10, 5))
sns.barplot(x=closed_lost_reasons.values, y=closed_lost_reasons.index, palette="Reds")
plt.xlabel("Number of Deals Lost")
plt.ylabel("Closed Lost Reason")
plt.title("Top Reasons for Closed Lost Deals")
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure

##########################################
# 10. Closed Won Revenue by Region (Seaborn)
##########################################
st.markdown("## Closed Won Revenue by Region (Seaborn)")
closed_won_revenue = closed_won_df.groupby("Region")["Total_Revenue"].sum().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(x=closed_won_revenue["Region"], y=closed_won_revenue["Total_Revenue"], palette="Purples")
plt.xlabel("Region")
plt.ylabel("Total Revenue (£)")
plt.title("Total Closed Won Revenue by Region")
plt.xticks(rotation=45)
st.pyplot(plt.gcf())
plt.clf()

##########################################
# 11. Sales Cycle Distribution by Region (Seaborn Boxplot)
##########################################
st.markdown("## Sales Cycle Distribution by Region (Seaborn)")
plt.figure(figsize=(10, 5))
sns.boxplot(x=df[df["Deal_Stage"] == "Closed Won"]["Region"], 
            y=df[df["Deal_Stage"] == "Closed Won"]["Sales_Cycle_Days"],
            palette="coolwarm")
plt.xlabel("Region")
plt.ylabel("Sales Cycle Duration (Days)")
plt.title("Distribution of Sales Cycle by Region")
st.pyplot(plt.gcf())
plt.clf()

##########################################
# 12. Pipeline Breakdown Stacked Bar Chart (Plotly)
##########################################
st.markdown("## Pipeline Breakdown by Region & Segment (Stacked Bar Chart)")

# Prepare the stacked bar data
stacked_bar_data = (
    df.groupby(["Region", "Segment", "Deal_Stage"])
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
# Adjust layout: remove extra x-axis labels in facets
fig_stacked.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis_title="Pipeline Stage"
)
# Remove the "Region=" prefix from facet annotations
for annotation in fig_stacked.layout.annotations:
    annotation.text = annotation.text.split("=")[-1]
st.plotly_chart(fig_stacked, use_container_width=True)

##########################################
# 13. Lead Analysis
##########################################
st.markdown("## Lead Analysis")

# Aggregate lead volume overview
lead_volume_overview = (
    lead_df.groupby(["Region", "Segment", "Lead_Source"])
    .agg(Total_Leads=("Lead_ID", "count"))
    .reset_index()
)
st.write("**Lead Volume Overview:**")
st.dataframe(lead_volume_overview)

# Plot leads by Region & Source with facets by Segment
fig_region = px.bar(
    lead_volume_overview,
    x="Region",
    y="Total_Leads",
    color="Lead_Source",
    barmode="stack",
    facet_col="Segment",
    title="Leads by Region & Source",
    labels={"Total_Leads": "Total Leads", "Region": "Region", "Lead_Source": "Lead Source"}
)
st.plotly_chart(fig_region, use_container_width=True)

# Plot leads by Segment & Source with facets by Region
fig_segment = px.bar(
    lead_volume_overview,
    x="Segment",
    y="Total_Leads",
    color="Lead_Source",
    barmode="stack",
    facet_col="Region",
    title="Leads by Segment & Source",
    labels={"Total_Leads": "Total Leads", "Segment": "Segment", "Lead_Source": "Lead Source"}
)
st.plotly_chart(fig_segment, use_container_width=True)

# Sunburst chart for a hierarchical view of leads
fig_sunburst = px.sunburst(
    lead_volume_overview,
    path=["Region", "Segment", "Lead_Source"],
    values="Total_Leads",
    title="Hierarchical View of Lead Volume",
    labels={"Total_Leads": "Total Leads", "Region": "Region", "Segment": "Segment", "Lead_Source": "Lead Source"}
)
st.plotly_chart(fig_sunburst, use_container_width=True)

##########################################
# 14. Campaign Type Analysis
##########################################
st.markdown("## Campaign Type Analysis")

# Filter for Marketing Campaign leads and prepare date fields
campaign_type_analysis = lead_df[lead_df["Lead_Source"] == "Marketing Campaign"].copy()
campaign_type_analysis["Lead_Creation_Date"] = pd.to_datetime(campaign_type_analysis["Lead_Creation_Date"])
# Create month period (as string for Plotly)
campaign_type_analysis["Lead_Creation_Month"] = campaign_type_analysis["Lead_Creation_Date"].dt.to_period("M").astype(str)
# Create flag for conversion
campaign_type_analysis["Converted_Flag"] = np.where(
    campaign_type_analysis["Converted_to_Opportunity"] == "Yes", 1, 0
)

# Campaign type condensed metrics
campaign_type_condensed = (
    campaign_type_analysis.groupby("Campaign_Type")
    .agg(
        Total_Leads=("Lead_ID", "count"),
        Total_Opportunities=("Converted_Flag", "sum"),
        Average_Conversion_Time=("Time_to_Conversion (days)", "mean")
    )
    .reset_index()
)
campaign_type_condensed["Average_Conversion_Time"] = campaign_type_condensed["Average_Conversion_Time"].round()
st.write("**Campaign Type Condensed Metrics:**")
st.dataframe(campaign_type_condensed.head(12))

# Monthly lead trend by campaign type
monthly_campaign_trend = (
    campaign_type_analysis.groupby(["Lead_Creation_Month", "Campaign_Type"])
    .agg(Total_Leads=("Lead_ID", "count"))
    .reset_index()
)
fig_trend = px.line(
    monthly_campaign_trend,
    x="Lead_Creation_Month",
    y="Total_Leads",
    color="Campaign_Type",
    title="Monthly Lead Volume by Campaign Type",
    labels={"Total_Leads": "Total Leads", "Lead_Creation_Month": "Month"}
)
st.plotly_chart(fig_trend, use_container_width=True)

# Generate Opportunities Over Time
campaign_type_analysis["Opportunity_Creation_Month"] = campaign_type_analysis.apply(
    lambda row: row["Lead_Creation_Month"] if row["Converted_to_Opportunity"] == "Yes" else np.nan,
    axis=1
)
opportunity_trend = (
    campaign_type_analysis.dropna(subset=["Opportunity_Creation_Month"])
    .groupby("Opportunity_Creation_Month")
    .agg(Opportunities_Generated=("Lead_ID", "count"))
    .reset_index()
)
opportunity_trend["Opportunity_Creation_Month"] = opportunity_trend["Opportunity_Creation_Month"].astype(str)
fig_opportunity_trend = px.line(
    opportunity_trend,
    x="Opportunity_Creation_Month",
    y="Opportunities_Generated",
    title="Opportunities Generated Over Time",
    labels={"Opportunities_Generated": "Opportunities Generated", "Opportunity_Creation_Month": "Month"}
)
st.plotly_chart(fig_opportunity_trend, use_container_width=True)

# Conversion rate by Segment & Campaign Type
conversion_by_segment = (
    campaign_type_analysis.groupby(["Segment", "Campaign_Type"])
    .agg(Total_Leads=("Lead_ID", "count"), Converted_Leads=("Converted_Flag", "sum"))
    .reset_index()
)
conversion_by_segment["Conversion_Rate"] = (conversion_by_segment["Converted_Leads"] / conversion_by_segment["Total_Leads"]) * 100
fig_conversion = px.bar(
    conversion_by_segment,
    x="Segment",
    y="Conversion_Rate",
    color="Campaign_Type",
    barmode="group",
    title="Conversion Rate by Campaign Type & Segment",
    labels={"Conversion_Rate": "Conversion Rate (%)", "Segment": "Segment", "Campaign_Type": "Campaign Type"}
)
st.plotly_chart(fig_conversion, use_container_width=True)

# Lead Distribution by Segment & Source
lead_by_segment = (
    lead_df.groupby(["Segment", "Lead_Source"])
    .agg(Total_Leads=("Lead_ID", "count"))
    .reset_index()
)
fig_lead_segment = px.bar(
    lead_by_segment,
    x="Segment",
    y="Total_Leads",
    color="Lead_Source",
    barmode="stack",
    title="Lead Distribution by Segment & Source",
    labels={"Total_Leads": "Total Leads", "Segment": "Segment", "Lead_Source": "Lead Source"}
)
st.plotly_chart(fig_lead_segment, use_container_width=True)

# Conversion rate heatmap by Region & Segment
conversion_by_region_segment = (
    campaign_type_analysis.groupby(["Region", "Segment"])
    .agg(Total_Leads=("Lead_ID", "count"), Converted_Leads=("Converted_Flag", "sum"))
    .reset_index()
)
conversion_by_region_segment["Conversion_Rate"] = (
    conversion_by_region_segment["Converted_Leads"] / conversion_by_region_segment["Total_Leads"]
) * 100
heatmap_conv_data = conversion_by_region_segment.pivot(index="Region", columns="Segment", values="Conversion_Rate")
fig_conversion_heatmap = px.imshow(
    heatmap_conv_data,
    labels=dict(color="Conversion Rate (%)"),
    title="Lead Conversion Rate by Region & Segment",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_conversion_heatmap, use_container_width=True)

##########################################
# 15. Customer Counts by Region
##########################################
st.markdown("## Customer Counts by Region")
customer_counts_by_region = (
    new_customers.groupby("Region")["Opportunity_ID"].count() +
    renewals.groupby("Region")["Opportunity_ID"].count() -
    churned_customers.groupby("Region")["Opportunity_ID"].count()
).reset_index(name="Total_Customers")
st.dataframe(customer_counts_by_region)

##########################################
# End of Dashboard
##########################################
st.markdown("### End of Dashboard")
