import streamlit as st
import pandas as pd
import locale

# Configure locale for currency formatting
locale.setlocale(locale.LC_ALL, '')

def format_currency(value):
    """Format numbers as currency."""
    try:
        return locale.currency(value, grouping=True)
    except:
        return f"${value:,.2f}"

def format_input_as_currency(input_value):
    """Format input strings as currency."""
    try:
        if not input_value:
            return ""
        input_value = input_value.replace(",", "").replace("$", "")
        value = float(input_value)
        return f"${value:,.2f}"
    except ValueError:
        return ""

# Streamlit App Start
st.title("Tuition and Expense Planning Tool")

# Step 1: Previous Year's Total Expenses
st.subheader("Step 1: Previous Year's Total Expenses")
previous_expenses_input = st.text_input("Enter the Previous Year's Total Expenses ($)", "")
formatted_previous_expenses = format_input_as_currency(previous_expenses_input)
try:
    previous_expenses = float(formatted_previous_expenses.replace(",", "").replace("$", ""))
except ValueError:
    previous_expenses = 0.0

if previous_expenses > 0:
    st.success(f"Previous Year's Total Expenses: {format_currency(previous_expenses)}")
else:
    st.warning("Please enter valid total expenses.")

# Step 2: Operational Tuition Increase (OTI) Calculation
st.subheader("Step 2: Operational Tuition Increase (OTI)")
roi = st.number_input("Rate of Inflation (ROI) %", min_value=0.0, step=0.01, value=3.32)
rpi = st.number_input("Rate of Productivity Increase (RPI) %", min_value=0.0, step=0.01, value=2.08)
oti_percentage = roi + rpi
st.info(f"Operational Tuition Increase (OTI): {oti_percentage:.2f}%")

# Step 3: Strategic Items (SI) Input
st.subheader("Step 3: Strategic Items (SI)")
strategic_items = []
num_items = st.number_input("Number of Strategic Items", min_value=0, max_value=10, step=1, value=0)
for i in range(num_items):
    item_name = st.text_input(f"Strategic Item {i+1} Name", f"Item {i+1}")
    item_cost_input = st.text_input(f"Cost of {item_name} ($)", "")
    formatted_cost = format_input_as_currency(item_cost_input)
    try:
        item_cost = float(formatted_cost.replace(",", "").replace("$", ""))
    except ValueError:
        item_cost = 0.0
    item_description = st.text_area(f"Description of {item_name}", f"Details for {item_name}")
    strategic_items.append({"Item": item_name, "Cost": item_cost, "Description": item_description})

total_si_cost = sum(item["Cost"] for item in strategic_items)
si_percentage = (total_si_cost / previous_expenses) * 100 if previous_expenses > 0 else 0.0
st.info(f"Total Strategic Items Cost: {format_currency(total_si_cost)}")
st.info(f"Strategic Items (SI) Percentage: {si_percentage:.2f}%")

# Display Strategic Items Table
if strategic_items:
    st.subheader("Strategic Items Overview")
    strategic_items_df = pd.DataFrame(strategic_items)
    st.table(strategic_items_df)

# Step 4: Total Expense Growth and Budget Projection
st.subheader("Step 4: Total Expense Growth and Budget Projection")
total_increase_percentage = oti_percentage + si_percentage
new_expense_budget = previous_expenses * (1 + total_increase_percentage / 100)

st.write(f"Total Increase in Expenses: {total_increase_percentage:.2f}%")
st.write(f"Projected New Expense Budget: {format_currency(new_expense_budget)}")

# Step 5: Tuition Assistance
st.subheader("Step 5: Tuition Assistance")
financial_aid_input = st.text_input("Total Financial Aid Provided ($)", "")
formatted_financial_aid = format_input_as_currency(financial_aid_input)
try:
    financial_aid = float(formatted_financial_aid.replace(",", "").replace("$", ""))
except ValueError:
    financial_aid = 0.0

if financial_aid > 0:
    st.success(f"Total Financial Aid: {format_currency(financial_aid)}")
else:
    st.warning("Please enter valid financial aid amount.")

# Step 6: Tuition Adjustment by Grade Level
st.subheader("Step 6: Tuition Adjustment by Grade Level")

# Grade-level data input
num_grades = st.number_input("Number of Grade Levels", min_value=1, max_value=12, step=1, value=1)
grades = []
for i in range(num_grades):
    grade_name = st.text_input(f"Grade Level {i+1} Name", f"Grade {i+1}")
    num_students = st.number_input(f"Number of Students in {grade_name}", min_value=0, step=1, value=0)
    current_tuition = st.number_input(f"Current Tuition per Student in {grade_name} ($)", min_value=0.0, step=0.01, value=0.0)
    grades.append({"Grade": grade_name, "Number of Students": num_students, "Current Tuition": current_tuition})

# Calculate initial projected tuition increases
grades_df = pd.DataFrame(grades)
grades_df["Projected Tuition per Student"] = grades_df["Current Tuition"] * (1 + total_increase_percentage / 100)
grades_df["Total Current Tuition"] = grades_df["Number of Students"] * grades_df["Current Tuition"]
grades_df["Total Projected Tuition"] = grades_df["Number of Students"] * grades_df["Projected Tuition per Student"]

# Initialize current_total_tuition before further use
current_total_tuition = grades_df["Total Current Tuition"].sum()

# Display initial results
if st.button("View Results"):
    st.subheader("Initial Projected Tuition Increase")
    st.table(grades_df)

    # Real-time metrics after results
    projected_total_tuition = grades_df["Total Projected Tuition"].sum()
    tuition_assistance_ratio_projected = (financial_aid / projected_total_tuition) * 100 if projected_total_tuition > 0 else 0.0
    income_to_expense_ratio_projected = (projected_total_tuition / new_expense_budget) * 100 if new_expense_budget > 0 else 0.0
    tuition_rate_increase_projected = ((projected_total_tuition - current_total_tuition) / current_total_tuition) * 100 if current_total_tuition > 0 else 0.0

    # Projected Results with Explanations
    st.subheader("Projected Results")
    st.write(f"**Current Total Tuition:** {format_currency(current_total_tuition)}")
    st.write("This represents the total tuition revenue currently collected across all grades. It’s calculated by multiplying the number of students by the current tuition rates for each grade.")

    st.write(f"**Projected Total Tuition (Initial Increase):** {format_currency(projected_total_tuition)}")
    st.write("This reflects the estimated revenue after applying the calculated percentage increase (from inflation, productivity, and strategic items) to the current tuition rates.")

    st.write(f"**Projected Tuition Assistance Ratio:** {tuition_assistance_ratio_projected:.2f}%")
    st.write("This measures how much of the tuition revenue will be allocated to financial aid. It helps ensure financial aid remains proportional to the new revenue levels.")

    st.write(f"**Projected Income to Expense (I/E) Ratio:** {income_to_expense_ratio_projected:.2f}%")
    st.write("This measures whether projected tuition revenue is enough to cover the school’s expenses. A ratio above 100% means revenue exceeds expenses, while below 100% means a shortfall.")

    st.write(f"**Tuition Rate Increase (Projected):** {tuition_rate_increase_projected:.2f}%")
    st.write("This represents the percentage growth in revenue due to the projected tuition changes compared to current tuition levels.")

# Allow user to adjust tuition per grade level
st.subheader("Adjust Tuition by Grade Level")
adjusted_tuitions = []
for i, grade in grades_df.iterrows():
    adjusted_tuition = st.number_input(
        f"Adjusted Tuition for {grade['Grade']} ($)",
        min_value=0.0,
        step=0.01,
        value=grade["Projected Tuition per Student"],
    )
    adjusted_tuitions.append(adjusted_tuition)

grades_df["Adjusted Tuition per Student"] = adjusted_tuitions
grades_df["Total Adjusted Tuition"] = grades_df["Number of Students"] * grades_df["Adjusted Tuition per Student"]

# Real-time metrics for adjusted tuition
adjusted_total_tuition = grades_df["Total Adjusted Tuition"].sum()
tuition_assistance_ratio_adjusted = (financial_aid / adjusted_total_tuition) * 100 if adjusted_total_tuition > 0 else 0.0
income_to_expense_ratio_adjusted = (adjusted_total_tuition / new_expense_budget) * 100 if new_expense_budget > 0 else 0.0
tuition_rate_increase_adjusted = ((adjusted_total_tuition - current_total_tuition) / current_total_tuition) * 100 if current_total_tuition > 0 else 0.0

# Adjusted Results with Explanations
st.subheader("Adjusted Results")
st.write(f"**Adjusted Total Tuition (User Adjusted):** {format_currency(adjusted_total_tuition)}")
st.write("This is the revenue collected based on user-defined adjustments to tuition rates for each grade.")

st.write(f"**Adjusted Tuition Assistance Ratio:** {tuition_assistance_ratio_adjusted:.2f}%")
st.write("This measures how much of the adjusted tuition revenue is allocated to financial aid.")

st.write(f"**Adjusted Income to Expense (I/E) Ratio:** {income_to_expense_ratio_adjusted:.2f}%")
st.write("This shows whether adjusted tuition revenue is sufficient to cover the school’s expenses after adjustments.")

st.write(f"**Tuition Rate Increase (Adjusted):** {tuition_rate_increase_adjusted:.2f}%")
st.write("This shows the percentage increase in tuition revenue based on the user’s adjustments.")
