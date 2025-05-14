import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load model using correct path
try:
    model_path = os.path.join(current_dir, 'final_sales_forecasting_model.pkl')
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.info("Please ensure all required model files are in the correct location.")
    st.stop()

st.set_page_config(page_title="Store Sales Predictor", layout="centered")
st.title("📦 Store Sales Prediction App")

with st.form("prediction_form"):
    st.header("Basic Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic order information
        ship_mode = st.selectbox("Ship Mode", 
            ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
        segment = st.selectbox("Segment", 
            ['Consumer', 'Corporate', 'Home Office'])
        
        # Location information
        city = st.selectbox("City", 
            ['Los Angeles', 'New York', 'Seattle', 'San Francisco', 'Chicago'])
        state = st.selectbox("State", 
            ['California', 'New York', 'Washington', 'Illinois', 'Texas'])
        region = st.selectbox("Region", 
            ['West', 'East', 'Central', 'South'])

    with col2:
        # Product information
        category = st.selectbox("Category", 
            ['Furniture', 'Office Supplies', 'Technology'])
        sub_category = st.selectbox("Sub-Category", 
            ['Chairs', 'Tables', 'Bookcases', 'Phones', 'Storage', 
             'Furnishings', 'Art', 'Labels', 'Binders', 'Accessories'])
        
        # Order details
        quantity = st.number_input("Quantity", min_value=1, max_value=100, value=3)
        discount = st.slider("Discount", 0.0, 1.0, 0.2, 0.05)
        profit = st.number_input("Profit", value=50.0, min_value=-1000.0, max_value=1000.0)

    st.header("Order Timing")
    col3, col4 = st.columns(2)
    
    with col3:
        order_date = st.date_input("Order Date", value=datetime.now())
        ship_date = st.date_input("Ship Date", value=datetime.now())
        
    with col4:
        # These will be calculated based on the dates
        order_day = order_date.weekday()
        order_month = order_date.month
        order_year = order_date.year
        order_quarter = (order_month - 1) // 3 + 1
        is_weekend = 1 if order_day >= 5 else 0
        days_to_ship = (ship_date - order_date).days
        
        # Display calculated values
        st.write("Calculated Order Information:")
        st.write(f"Day of Week: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][order_day]}")
        st.write(f"Month: {order_month}")
        st.write(f"Year: {order_year}")
        st.write(f"Quarter: {order_quarter}")
        st.write(f"Is Weekend: {'Yes' if is_weekend else 'No'}")
        st.write(f"Days to Ship: {days_to_ship}")

    st.header("Historical Metrics")
    col5, col6 = st.columns(2)
    
    with col5:
        total_quantity_ordered = st.number_input("Total Quantity Ordered", value=100.0)
        total_sales_by_product = st.number_input("Total Sales by Product", value=5000.0)
        avg_profit_by_product = st.number_input("Average Profit by Product", value=250.0)
        total_orders_by_customer = st.number_input("Total Orders by Customer", value=5)

    with col6:
        avg_order_value_customer = st.number_input("Avg Order Value (Customer)", value=1000.0)
        total_sales_by_customer = st.number_input("Total Sales by Customer", value=5000.0)
        region_total_sales = st.number_input("Region Total Sales", value=100000.0)
        region_avg_sales = st.number_input("Region Average Sales", value=2000.0)

    submitted = st.form_submit_button("Predict Sales")

if submitted:
    try:
        # Create input data with exact features from training
        input_data = pd.DataFrame({
            'Ship Mode': [ship_mode],
            'Segment': [segment],
            'City': [city],
            'State': [state],
            'Region': [region],
            'Category': [category],
            'Sub-Category': [sub_category],
            'Quantity': [quantity],
            'Discount': [discount],
            'Profit': [profit],
            'Order_DayOfWeek': [order_day],
            'Order_Month': [order_month],
            'Order_Year': [order_year],
            'Order_Quarter': [order_quarter],
            'Is_Weekend': [is_weekend],
            'Days_to_Ship': [days_to_ship],
            'total_quantity_ordered': [total_quantity_ordered],
            'total_sales_by_product': [total_sales_by_product],
            'avg_profit_by_product': [avg_profit_by_product],
            'total_orders_by_customer': [total_orders_by_customer],
            'avg_order_value_customer': [avg_order_value_customer],
            'total_sales_by_customer': [total_sales_by_customer],
            'region_total_sales': [region_total_sales],
            'region_avg_sales': [region_avg_sales]
        })

        # Apply label encoding to categorical columns
        categorical_features = ['Ship Mode', 'Segment', 'City', 'State', 'Region', 'Category', 'Sub-Category']
        
        le = LabelEncoder()
        for col in categorical_features:
            input_data[col] = le.fit_transform(input_data[col])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.success(f"📈 Predicted Sales: ${prediction:.2f}")
        
        # Display input summary
        st.subheader("Input Summary")
        
        # Display order details
        st.write("Order Details:")
        order_details = {
            "Category": category,
            "Sub-Category": sub_category,
            "Quantity": quantity,
            "Discount": f"{discount:.1%}",
            "Profit": f"${profit:.2f}"
        }
        st.write(order_details)
        
        # Display timing details
        st.write("Timing Details:")
        timing_details = {
            "Order Date": order_date.strftime("%Y-%m-%d"),
            "Ship Date": ship_date.strftime("%Y-%m-%d"),
            "Days to Ship": days_to_ship,
            "Quarter": order_quarter,
            "Is Weekend": "Yes" if is_weekend else "No"
        }
        st.write(timing_details)
        
        # Display location details
        st.write("Location Details:")
        location_details = {
            "City": city,
            "State": state,
            "Region": region,
            "Ship Mode": ship_mode,
            "Segment": segment
        }
        st.write(location_details)
        
        # Display debug information
        st.write("Debug Information:")
        st.write("Features used:", input_data.columns.tolist())
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug information:")
        st.write("Input data columns:", input_data.columns.tolist())
        st.info("Please check if all input values are within expected ranges.")