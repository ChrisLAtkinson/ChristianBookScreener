import streamlit as st
import locale

# Configure locale to display currency with commas and two decimal places
locale.setlocale(locale.LC_ALL, '')

# Helper function to format numbers as currency
def format_currency(value):
    return locale.currency(value, grouping=True)

# Title of the app
st.title("Tuition Calculation Tool")

# Step 1: Enter a Custom Title for the Report
report_title = st.text_input("Enter a Custom Title for the Report", "2025-26 Tuition Projection")

st.subheader("Step 2: Add Custom Grade Levels and Tuition Rates")
grades = []
num_students = []
current_tuition = []
num_grades = st.number_input("Number of Grade Levels", min_value=1, max_value=12, value=1)

for i in range(num_grades):
    grade = st.text_input(f"Grade Level {i+1} Name", f"Grade {i+1}")
    students = st.number_input(f"Number of Students in {grade}", min_value=1)
    tuition = st.number_input(f"Current Tuition per Student in {grade} ($)", min_value=0.0)
    
    grades.append(grade)
    num_students.append(students)
    current_tuition.append(tuition)

# Step 3: Add Strategic Items
st.subheader("Step 3: Add Strategic Items")
strategic_items = []
num_items = st.number_input("Number of Strategic Items", min_value=0, max_value=10, value=0)

for i in range(num_items):
    item_name = st.text_input(f"Strategic Item {i+1} Name", f"Item {i+1}")
    item_cost = st.number_input(f"Cost of {item_name} ($)", min_value=0.0)
    strategic_items.append(item_cost)

# Step 4: Previous Year's Expense Budget
st.subheader("Step 4: Enter the Previous Year's Total Expenses")
previous_expenses = st.number_input("Total Expenses ($)", min_value=0.0)

# Step 5: Operations Tuition Increase (OTI) Calculation
st.subheader("Step 5: Operations Tuition Increase (OTI) Calculation")
roi_percentage = st.number_input("Rate of Inflation (ROI) %", min_value=0.0, step=0.01)
efficiency_rate = 2.08 / 100  # 2.08% fixed rate of efficiency

# Step 6: Compensation Percentage
st.subheader("Step 6: Enter Compensation as a Percentage of Expenses")
compensation_percentage = st.number_input("Compensation Percentage (%)", min_value=0.0, max_value=100.0, value=70.0)

# Step 7: Financial Aid (Tuition Assistance) Calculation
st.subheader("Step 7: Financial Aid (Tuition Assistance)")
financial_aid = st.number_input("Total Financial Aid ($)", min_value=0.0)

# Calculate new tuition with average increase
if st.button("Calculate New Tuition"):
    total_current_tuition = sum([students * tuition for students, tuition in zip(num_students, current_tuition)])
    total_students = sum(num_students)
    total_strategic_costs = sum(strategic_items)
    
    adjusted_inflation = roi_percentage / 100 + efficiency_rate
    total_new_tuition = total_current_tuition + (total_current_tuition * adjusted_inflation) + total_strategic_costs
    avg_increase_percentage = ((total_new_tuition - total_current_tuition) / total_current_tuition) * 100
    
    # Calculate tuition assistance ratio
    tuition_assistance_ratio = (financial_aid / total_new_tuition) * 100 if total_new_tuition > 0 else 0

    # Display Results
    st.subheader("Results")
    st.write(f"Report Title: {report_title}")
    st.write(f"Total Current Tuition: {format_currency(total_current_tuition)}")
    st.write(f"Total New Tuition (with average increase): {format_currency(total_new_tuition)}")
    st.write(f"Average Tuition Increase Percentage: {avg_increase_percentage:.2f}%")
    st.write(f"Tuition Assistance Ratio: {tuition_assistance_ratio:.2f}%")
    
    # Display detailed grade-level results
    st.subheader("Tuition by Grade Level")
    for grade, students, tuition in zip(grades, num_students, current_tuition):
        new_tuition = tuition * (1 + avg_increase_percentage / 100)
        st.write(f"{grade}: Current Tuition: {format_currency(tuition)}, New Tuition: {format_currency(new_tuition)}, Increase: {format_currency(new_tuition - tuition)}")
