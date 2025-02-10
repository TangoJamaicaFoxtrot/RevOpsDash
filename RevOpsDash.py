import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

df = pd.read_csv("revops_data.csv")
df["Region"] = df["Region"].fillna("NA")
df["Deal_Size (£)"] = df["Deal_Size (£)"].round(2)
df.head()




updated_sales_data = pd.read_csv("updated_sales_data_balanced.csv")


# Filter for New Customers (Closed Won & Opportunity Type = "New Customer")
new_customers = df[
    (df["Deal_Stage"] == "Closed Won") & 
    (df["Opportunity_Type"] == "New Customer")
]

# Filter for Renewals (Closed Won & Opportunity Type = "Renewal")
renewals = df[
    (df["Deal_Stage"] == "Closed Won") & 
    (df["Opportunity_Type"] == "Renewal")
]

# Filter for Churned Customers (Closed Lost & Opportunity Type = "Renewal")
churned_customers = df[
    (df["Deal_Stage"] == "Closed Lost") & 
    (df["Opportunity_Type"] == "Renewal")
]

# Total Active Customers at a Fixed Date
total_customers = new_customers.shape[0] + renewals.shape[0] - churned_customers.shape[0]

# Print results
print(f"Total Customers: {total_customers}")
print(f"New Customers: {new_customers.shape[0]}")
print(f"Renewals: {renewals.shape[0]}")
print(f"Churned Customers: {churned_customers.shape[0]}")


# Count Closed Won and Closed Lost deals
win_loss_counts = df["Deal_Stage"].value_counts()

# Calculate Win Rate (Closed Won / (Closed Won + Closed Lost))
win_rate = (win_loss_counts.get("Closed Won", 0) / 
            (win_loss_counts.get("Closed Won", 0) + win_loss_counts.get("Closed Lost", 0))) * 100

print(f"Overall Win Rate: {win_rate:.2f}%")
# Total deals per Region
total_deals_by_region = df.groupby("Region")["Opportunity_ID"].count()

# Closed Won deals per Region
closed_won_by_region = df[df["Deal_Stage"] == "Closed Won"].groupby("Region")["Opportunity_ID"].count()

# Calculate Win Rate
win_rate_by_region = (closed_won_by_region / total_deals_by_region * 100).fillna(0)

print("\nWin Rate by Region:\n", win_rate_by_region)

# Total deals per Segment
total_deals_by_segment = df.groupby("Segment")["Opportunity_ID"].count()

# Closed Won deals per Segment
closed_won_by_segment = df[df["Deal_Stage"] == "Closed Won"].groupby("Segment")["Opportunity_ID"].count()

# Calculate Win Rate
win_rate_by_segment = (closed_won_by_segment / total_deals_by_segment * 100).fillna(0)

print("\nWin Rate by Segment:\n", win_rate_by_segment)

# Revenue breakdown, filter by region, segment, opp type
closed_won_df = df.loc[df["Deal_Stage"] == "Closed Won"].groupby(
    ["Region", "Segment", "Opportunity_Type"]
).agg(
    Total_Revenue=("Deal_Size (£)", "sum"), 
    Closed_Won_Deals=("Opportunity_ID", "count")
)

closed_won_df.head()
closed_lost_df = df.loc[df["Deal_Stage"] == "Closed Lost"].groupby(
    ["Region", "Segment", "Opportunity_Type"]
).agg(
    Closed_Lost_Count=("Opportunity_ID", "count"),
    Closed_Lost_Value=("Deal_Size (£)", "sum")
)
closed_lost_df.head()

closed_lost_reasons = df[df["Deal_Stage"] == "Closed Lost"]["Closed_Lost_Reason"].value_counts()

display(closed_lost_reasons)
# Pipeline analysis, win rates, sales cycle length
pipe_summary = df.groupby(["Region","Deal_Stage"]).agg(
    Deal_Count=("Opportunity_ID", "count"),
    Deal_Value=("Deal_Size (£)", "sum")
)
pipe_summary.head(6)

sales_cycle_analysis = df[df["Deal_Stage"] == "Closed Won"].groupby(["Region","Segment"]).agg(
    Avg_Sales_Cycle_Days=("Sales_Cycle_Days", "mean"),
    Median_Sales_Cycle_Days=("Sales_Cycle_Days", "median")
)

display(sales_cycle_analysis)
salesperson_leaderboards = df.groupby(["Salesperson", "Deal_Stage"]).agg(
    Stage_Value=("Deal_Size (£)", "sum"),
    Stage_Count=("Opportunity_ID", "count")
)

salesperson_leaderboards.head(15)   


import plotly.express as px

# Recalculate Win Rates
total_deals = df.groupby(["Region", "Segment"])["Opportunity_ID"].count()
closed_won = df[df["Deal_Stage"] == "Closed Won"].groupby(["Region", "Segment"])["Opportunity_ID"].count()

# Compute win rates per Region & Segment
win_rate = (closed_won / total_deals * 100).fillna(0).reset_index()
win_rate.columns = ["Region", "Segment", "Win Rate (%)"]

# Round win rates to two decimal places
win_rate["Win Rate (%)"] = win_rate["Win Rate (%)"].round(2)

# Pivot data for heatmap format
heatmap_data = win_rate.pivot(index="Segment", columns="Region", values="Win Rate (%)")

# Create heatmap using imshow (manual annotations for rounded values)
fig = px.imshow(
    heatmap_data,
    color_continuous_scale="Blues",
    title="Win Rate Heatmap by Region & Segment",
    labels={"color": "Win Rate (%)"},  # Remove x and y axis labels
)

# Manually add rounded text labels
fig.update_traces(
    text=heatmap_data.applymap(lambda x: f"{x:.2f}"),  # Format numbers
    texttemplate="%{text}"
)

# Adjust layout for better readability and increased width
fig.update_layout(
    width=1000,  # Increase width for better spacing
    height=500,  # Adjust height proportionally
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis_title="",  # Remove x-axis label
    yaxis_title="",  # Remove y-axis label
    coloraxis_colorbar=dict(title="Win Rate (%)")
)

# Show the heatmap
fig.show()
import plotly.express as px
import numpy as np

# Recalculate Win Rates
total_deals = df.groupby(["Region", "Segment"])["Opportunity_ID"].count()
closed_won = df[df["Deal_Stage"] == "Closed Won"].groupby(["Region", "Segment"])["Opportunity_ID"].count()

# Compute win rates per Region & Segment
win_rate = (closed_won / total_deals * 100).fillna(0).reset_index()
win_rate.columns = ["Region", "Segment", "Win Rate (%)"]

# Round win rates to two decimal places
win_rate["Win Rate (%)"] = win_rate["Win Rate (%)"].round(2)

# Pivot data for heatmap format
heatmap_data = win_rate.pivot(index="Segment", columns="Region", values="Win Rate (%)")

# Ensure dtype is correct (avoid NumPy bool issue)
heatmap_data = heatmap_data.astype(float)

# Create heatmap using imshow (manual annotations for rounded values)
fig = px.imshow(
    heatmap_data,
    color_continuous_scale="Blues",
    title="Win Rate Heatmap by Region & Segment",
    labels={"color": "Win Rate (%)"},  # Remove x and y axis labels
)

# Manually add rounded text labels
fig.update_traces(
    text=heatmap_data.applymap(lambda x: f"{x:.2f}"),  # Format numbers
    texttemplate="%{text}"
)

# Adjust layout for better readability and increased width
fig.update_layout(
    width=1300,  # Increase width further for better spacing
    height=500,  # Adjust height proportionally
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis_title="",  # Remove x-axis label
    yaxis_title="",  # Remove y-axis label
    coloraxis_colorbar=dict(title="Win Rate (%)")
)

# Show the heatmap
fig.show()

import plotly.express as px
import pandas as pd

# Recalculate Win Rates
total_deals_by_region = df.groupby("Region")["Opportunity_ID"].count()
closed_won_by_region = df[df["Deal_Stage"] == "Closed Won"].groupby("Region")["Opportunity_ID"].count()
win_rate_by_region = (closed_won_by_region / total_deals_by_region * 100).fillna(0).reset_index()
win_rate_by_region.columns = ["Category", "Win Rate (%)"]
win_rate_by_region["Type"] = "Region"

total_deals_by_segment = df.groupby("Segment")["Opportunity_ID"].count()
closed_won_by_segment = df[df["Deal_Stage"] == "Closed Won"].groupby("Segment")["Opportunity_ID"].count()
win_rate_by_segment = (closed_won_by_segment / total_deals_by_segment * 100).fillna(0).reset_index()
win_rate_by_segment.columns = ["Category", "Win Rate (%)"]
win_rate_by_segment["Type"] = "Segment"

# Combine Region & Segment Win Rates
win_rate_data = pd.concat([win_rate_by_region, win_rate_by_segment])

# Determine y-axis limit
max_win_rate = win_rate_data["Win Rate (%)"].max()
y_axis_limit = min(100, max_win_rate + 10)  # Add buffer but cap at 100%

# Create an interactive Plotly bar chart
fig = px.bar(
    win_rate_data,
    x="Category",
    y="Win Rate (%)",
    color="Type",
    barmode="group",
    title="Win Rate by Region & Segment",
    labels={"Category": "Region / Segment", "Win Rate (%)": "Win Rate (%)"},
    color_discrete_map={"Region": "blue", "Segment": "green"}  # Assign colors
)

# Adjust y-axis limit
fig.update_layout(yaxis=dict(range=[0, y_axis_limit]))

# Show the interactive chart
fig.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=closed_lost_reasons.values, y=closed_lost_reasons.index, palette="Reds")
plt.xlabel("Number of Deals Lost")
plt.ylabel("Closed Lost Reason")
plt.title("Top Reasons for Closed Lost Deals")
plt.show()


closed_won_revenue = closed_won_df.groupby("Region")["Total_Revenue"].sum()

plt.figure(figsize=(10, 5))
sns.barplot(x=closed_won_revenue.index, y=closed_won_revenue.values, palette="Purples")
plt.xlabel("Region")
plt.ylabel("Total Revenue (£)")
plt.title("Total Closed Won Revenue by Region")
plt.xticks(rotation=45)
plt.show()



plt.figure(figsize=(10, 5))
sns.boxplot(x=df[df["Deal_Stage"] == "Closed Won"]["Region"], 
            y=df[df["Deal_Stage"] == "Closed Won"]["Sales_Cycle_Days"], palette="coolwarm")
plt.xlabel("Region")
plt.ylabel("Sales Cycle Duration (Days)")
plt.title("Distribution of Sales Cycle by Region")
plt.show()

import plotly.express as px

# Create the updated stacked bar chart
fig = px.bar(
    stacked_bar_data,
    x="Deal_Stage",
    y="Deal Count",
    color="Segment",  # Different colors for each Segment
    barmode="stack",  # Stack bars on top of each other
    facet_col="Region",  # Create a separate chart per Region
    title="Pipeline Breakdown by Region & Segment",
    labels={"Deal_Stage": "Pipeline Stage", "Deal Count": "Total Deals", "Segment": "Customer Segment"},
)

# Update layout to:
# - Remove x-axis labels from individual facets
# - Set a single global x-axis label
fig.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis_title="Pipeline Stage",  # Single global x-axis label
)

# Remove "Pipeline Stage" from individual facet axes
fig.for_each_xaxis(lambda axis: axis.update(title=""))

# Update facet labels to display only region names without "Region=" prefix
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Show the updated chart
fig.show()


# Lead Analysis

lead_df = pd.read_csv("revops_lead_data.csv")
lead_df["Region"] = lead_df["Region"].fillna("NA")
print(lead_df.head())
lead_volume_overview = lead_df.groupby(["Region", "Segment", "Lead_Source"]).agg(
    Total_Leads=("Lead_ID", "count")
)
display(lead_volume_overview)
import plotly.express as px

# Reset index for visualization
lead_volume_overview = lead_volume_overview.reset_index()

fig_region = px.bar(
    lead_volume_overview,
    x="Region",
    y="Total_Leads",
    color="Lead_Source",
    barmode="stack",
    facet_col="Segment",  # Allows filtering by Segment
    title="Leads by Region & Source",
    labels={"Total_Leads": "Total Leads", "Region": "Region", "Lead_Source": "Lead Source"},
)

fig_region.show()

fig_segment = px.bar(
    lead_volume_overview,
    x="Segment",
    y="Total_Leads",
    color="Lead_Source",
    barmode="stack",
    facet_col="Region",  # Allows filtering by Region
    title="Leads by Segment & Source",
    labels={"Total_Leads": "Total Leads", "Segment": "Segment", "Lead_Source": "Lead Source"},
)

fig_segment.show()

fig_sunburst = px.sunburst(
    lead_volume_overview,
    path=["Region", "Segment", "Lead_Source"],
    values="Total_Leads",
    title="Hierarchical View of Lead Volume",
    labels={"Total_Leads": "Total Leads", "Region": "Region", "Segment": "Segment", "Lead_Source": "Lead Source"},
)

fig_sunburst.show()

lead_df.head()
campaign_type_analysis = lead_df[lead_df["Lead_Source"] == "Marketing Campaign"]
campaign_type_analysis["Lead_Creation_Date"] = pd.to_datetime(campaign_type_analysis["Lead_Creation_Date"])
campaign_type_analysis["Lead_Creation_Month"] = campaign_type_analysis["Lead_Creation_Date"].dt.to_period("M")

import numpy as np

campaign_type_analysis["Converted_Flag"] = np.where(
    campaign_type_analysis["Converted_to_Opportunity"] == "Yes", 1, 0
)

campaign_type_analysis.head()
campaign_type_condensed = campaign_type_analysis.groupby("Campaign_Type").agg(
    Total_Leads=("Lead_ID", "count"),
    Total_Opportunities=("Converted_Flag", "sum"),
    Average_Conversion_Time=("Time_to_Conversion (days)", "mean")
)
campaign_type_condensed["Average_Conversion_Time"] = campaign_type_condensed["Average_Conversion_Time"].round()
campaign_type_condensed.head(12)
# Convert Lead_Creation_Month to string format (YYYY-MM)
campaign_type_analysis["Lead_Creation_Month"] = campaign_type_analysis["Lead_Creation_Month"].astype(str)

# Aggregate leads by month and campaign type
monthly_campaign_trend = campaign_type_analysis.groupby(
    ["Lead_Creation_Month", "Campaign_Type"]
).agg(Total_Leads=("Lead_ID", "count")).reset_index()

# Create the Plotly line chart
fig_trend = px.line(
    monthly_campaign_trend,
    x="Lead_Creation_Month",
    y="Total_Leads",
    color="Campaign_Type",
    title="Monthly Lead Volume by Campaign Type",
    labels={"Total_Leads": "Total Leads", "Lead_Creation_Month": "Month"},
)

fig_trend.show()

# Create 'Opportunity Creation Month' for Converted Leads
campaign_type_analysis["Opportunity_Creation_Month"] = campaign_type_analysis.apply(
    lambda row: row["Lead_Creation_Month"] if row["Converted_to_Opportunity"] == "Yes" else np.nan, axis=1
)

# Drop NaN values (keep only converted leads)
opportunity_trend = campaign_type_analysis.dropna(subset=["Opportunity_Creation_Month"])

# Aggregate converted leads by month
opportunity_trend = opportunity_trend.groupby("Opportunity_Creation_Month").agg(
    Opportunities_Generated=("Lead_ID", "count")
).reset_index()

# Convert to string for Plotly compatibility
opportunity_trend["Opportunity_Creation_Month"] = opportunity_trend["Opportunity_Creation_Month"].astype(str)

# Create Line Chart for Opportunities Generated Over Time
fig_opportunity_trend = px.line(
    opportunity_trend,
    x="Opportunity_Creation_Month",
    y="Opportunities_Generated",
    title="Opportunities Generated Over Time",
    labels={"Opportunities_Generated": "Opportunities Generated", "Opportunity_Creation_Month": "Month"},
)

fig_opportunity_trend.show()

# Aggregate conversion rate by segment and campaign type
conversion_by_segment = campaign_type_analysis.groupby(["Segment", "Campaign_Type"]).agg(
    Total_Leads=("Lead_ID", "count"),
    Converted_Leads=("Converted_Flag", "sum")
).reset_index()

# Calculate conversion rate
conversion_by_segment["Conversion_Rate"] = (conversion_by_segment["Converted_Leads"] / conversion_by_segment["Total_Leads"]) * 100

# Create the Plotly bar chart
fig_conversion = px.bar(
    conversion_by_segment,
    x="Segment",
    y="Conversion_Rate",
    color="Campaign_Type",
    barmode="group",
    title="Conversion Rate by Campaign Type & Segment",
    labels={"Conversion_Rate": "Conversion Rate (%)", "Segment": "Segment", "Campaign_Type": "Campaign Type"},
)

fig_conversion.show()

import plotly.express as px

# Aggregate lead counts by Segment & Lead Source
lead_by_segment = lead_df.groupby(["Segment", "Lead_Source"]).agg(
    Total_Leads=("Lead_ID", "count")
).reset_index()

# Create the bar chart
fig_lead_segment = px.bar(
    lead_by_segment,
    x="Segment",
    y="Total_Leads",
    color="Lead_Source",
    barmode="stack",
    title="Lead Distribution by Segment & Source",
    labels={"Total_Leads": "Total Leads", "Segment": "Segment", "Lead_Source": "Lead Source"},
)

fig_lead_segment.show()

# Aggregate conversion data by Region & Segment
conversion_by_region_segment = campaign_type_analysis.groupby(["Region", "Segment"]).agg(
    Total_Leads=("Lead_ID", "count"),
    Converted_Leads=("Converted_Flag", "sum")
).reset_index()

# Calculate conversion rate
conversion_by_region_segment["Conversion_Rate"] = (
    conversion_by_region_segment["Converted_Leads"] / conversion_by_region_segment["Total_Leads"]
) * 100

# Create the heatmap
fig_conversion_heatmap = px.imshow(
    conversion_by_region_segment.pivot(index="Region", columns="Segment", values="Conversion_Rate"),
    labels=dict(color="Conversion Rate (%)"),
    title="Lead Conversion Rate by Region & Segment",
    color_continuous_scale="Blues"
)

fig_conversion_heatmap.show()

# Count customers by region
customer_counts_by_region = (
    new_customers.groupby("Region")["Opportunity_ID"].count() +
    renewals.groupby("Region")["Opportunity_ID"].count() -
    churned_customers.groupby("Region")["Opportunity_ID"].count()
).reset_index(name="Total_Customers")

# Display results
display(customer_counts_by_region)

df.head()



