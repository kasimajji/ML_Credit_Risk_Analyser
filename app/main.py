import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from prediction_helper import predict

# Set page configuration with a modern theme
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {font-size:2.5rem; color:#1E88E5; font-weight:bold; text-align:center; margin-bottom:1rem;}
    .sub-header {font-size:1.5rem; color:#26A69A; font-weight:bold; margin-top:1.5rem;}
    .card {background-color:#f8f9fa; border-radius:10px; padding:1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .result-card {background-color:#e8f5e9; border-radius:10px; padding:1.5rem; margin-top:1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .warning-card {background-color:#fff3e0; border-radius:10px; padding:1.5rem; margin-top:1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .info-text {color:#546E7A; font-size:0.9rem;}
    .highlight {color:#1E88E5; font-weight:bold;}
    .result-text {font-size:1.2rem; font-weight:bold;}
    .footer {text-align:center; margin-top:3rem; color:#78909C; font-size:0.8rem;}
    .stButton>button {background-color:#1E88E5; color:white; font-weight:bold; border-radius:5px;}
    .stButton>button:hover {background-color:#1565C0;}
</style>
""", unsafe_allow_html=True)

# Animated header with progress bar for visual appeal
with st.container():
    st.markdown('<div class="main-header">Credit Risk Analyzer</div>', unsafe_allow_html=True)
    
    # Animated progress bar for visual effect
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.005)
    progress_bar.empty()

# Create sidebar for additional information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/financial-analytics.png")
    st.markdown('<div class="sub-header">Internal Credit Risk Assessment Tool</div>', unsafe_allow_html=True)
    st.markdown("""
    This tool provides credit risk assessment using the machine learning model developed by the data science team.
    
    **Model Information:**
    - Algorithm: Logistic Regression
    - Training Data: 37,500 loan records
    - Features: 19 financial and demographic variables
    - Accuracy: 92.4% on test data
    """)
    
    st.markdown('<div class="sub-header">Key Metrics</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Credit Score**: 300-900 scale
    - **Default Probability**: Statistical likelihood of default
    - **Rating**: Internal risk classification
    """)

# Initialize session state variables
if 'tab_index' not in st.session_state:
    st.session_state.tab_index = 0
if 'results_calculated' not in st.session_state:
    st.session_state.results_calculated = False
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
    
# Initialize input variables in session state to make them accessible across tabs
if 'input_data' not in st.session_state:
    st.session_state.input_data = {
        'age': 28,
        'income': 1200000,
        'loan_amount': 2560000,
        'loan_tenure_months': 36,
        'avg_dpd_per_delinquency': 20,
        'delinquency_ratio': 30,
        'credit_utilization_ratio': 30,
        'num_open_accounts': 2,
        'residence_type': 'Owned',
        'loan_purpose': 'Education',
        'loan_type': 'Secured'
    }

# Function to change tab
def change_tab(tab_index):
    st.session_state.tab_index = tab_index
    st.session_state.form_submitted = True

# Create radio buttons for tabs but make them horizontal and look like tabs
tab_options = ["📋 Application Form", "📊 Results", "ℹ️ Help"]
selected_tab = st.radio("Select Tab", tab_options, index=st.session_state.tab_index, horizontal=True, label_visibility="collapsed")

# Application Form Tab
if selected_tab == "📋 Application Form":
    st.markdown('<div class="sub-header">Loan Application Details</div>', unsafe_allow_html=True)
    
    # Create two columns for personal and financial information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="highlight">Personal Information</p>', unsafe_allow_html=True)
        
        st.session_state.input_data['age'] = st.slider(
            'Age', 
            min_value=18, 
            max_value=100, 
            value=st.session_state.input_data['age'], 
            help="Applicant's age in years"
        )
        
        st.session_state.input_data['residence_type'] = st.selectbox(
            'Residence Type', 
            ['Owned', 'Rented', 'Mortgage'],
            index=['Owned', 'Rented', 'Mortgage'].index(st.session_state.input_data['residence_type']),
            help="Type of residence where applicant currently lives"
        )
        
        st.session_state.input_data['num_open_accounts'] = st.slider(
            'Number of Open Loan Accounts', 
            min_value=0, 
            max_value=10, 
            value=st.session_state.input_data['num_open_accounts'],
            help="Number of currently active loan accounts"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<p class="highlight">Financial Information</p>', unsafe_allow_html=True)
        
        st.session_state.input_data['income'] = st.number_input(
            'Annual Income (₹)', 
            min_value=0, 
            value=st.session_state.input_data['income'],
            step=50000,
            help="Annual income in Rupees"
        )
        
        st.session_state.input_data['loan_amount'] = st.number_input(
            'Loan Amount (₹)', 
            min_value=0, 
            value=st.session_state.input_data['loan_amount'],
            step=50000,
            help="Requested loan amount in Rupees"
        )
        
        st.session_state.input_data['loan_tenure_months'] = st.slider(
            'Loan Tenure (months)', 
            min_value=6, 
            max_value=120, 
            value=st.session_state.input_data['loan_tenure_months'],
            step=6,
            help="Duration of loan in months"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two more columns for credit history and loan details
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<p class="highlight">Credit History</p>', unsafe_allow_html=True)
        
        st.session_state.input_data['credit_utilization_ratio'] = st.slider(
            'Credit Utilization Ratio (%)', 
            min_value=0, 
            max_value=100, 
            value=st.session_state.input_data['credit_utilization_ratio'],
            help="Percentage of available credit currently being used"
        )
        
        st.session_state.input_data['delinquency_ratio'] = st.slider(
            'Delinquency Ratio (%)', 
            min_value=0, 
            max_value=100, 
            value=st.session_state.input_data['delinquency_ratio'],
            help="Percentage of loans with late payments"
        )
        
        st.session_state.input_data['avg_dpd_per_delinquency'] = st.slider(
            'Average Days Past Due', 
            min_value=0, 
            max_value=90, 
            value=st.session_state.input_data['avg_dpd_per_delinquency'],
            help="Average number of days payments are late"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<p class="highlight">Loan Details</p>', unsafe_allow_html=True)
        
        st.session_state.input_data['loan_purpose'] = st.selectbox(
            'Loan Purpose', 
            ['Education', 'Home', 'Auto', 'Personal'],
            index=['Education', 'Home', 'Auto', 'Personal'].index(st.session_state.input_data['loan_purpose']),
            help="Purpose for which the loan is being requested"
        )
        
        st.session_state.input_data['loan_type'] = st.selectbox(
            'Loan Type', 
            ['Secured', 'Unsecured'],
            index=['Secured', 'Unsecured'].index(st.session_state.input_data['loan_type']),
            help="Whether the loan is backed by collateral (secured) or not (unsecured)"
        )
        
        # Calculate and display loan to income ratio with animation
        loan_to_income_ratio = st.session_state.input_data['loan_amount'] / st.session_state.input_data['income'] if st.session_state.input_data['income'] > 0 else 0
        st.metric(
            "Loan to Income Ratio", 
            f"{loan_to_income_ratio:.2f}",
            delta="Higher is riskier" if loan_to_income_ratio > 2 else "Within safe limits",
            delta_color="inverse"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Use a unique key for the button to avoid conflicts
    if st.button('💰 Calculate Credit Risk', use_container_width=True, key="calc_risk_btn"):
        # Set the flags for results calculation and form submission
        st.session_state.results_calculated = True
        st.session_state.form_submitted = True
        # Change to the Results tab
        change_tab(1)
        # Rerun the app to apply the tab change
        st.rerun()

# Results Tab
elif selected_tab == "📊 Results":
    # Check if results were previously calculated
    if st.session_state.results_calculated:
        st.session_state.results_calculated = True
        
        # Show a spinner while calculating
        with st.spinner('Analyzing credit risk...'):
            # Call the predict function from the helper module
            probability, credit_score, rating = predict(
                st.session_state.input_data['age'], 
                st.session_state.input_data['income'], 
                st.session_state.input_data['loan_amount'], 
                st.session_state.input_data['loan_tenure_months'], 
                st.session_state.input_data['avg_dpd_per_delinquency'],
                st.session_state.input_data['delinquency_ratio'], 
                st.session_state.input_data['credit_utilization_ratio'], 
                st.session_state.input_data['num_open_accounts'],
                st.session_state.input_data['residence_type'], 
                st.session_state.input_data['loan_purpose'], 
                st.session_state.input_data['loan_type']
            )
            time.sleep(1)  # Simulate calculation time for better UX
        
        st.success('Analysis complete!')
        
        # Display results in a visually appealing way
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'<p class="highlight">Credit Score</p>', unsafe_allow_html=True)
            
            # Create a gauge chart for credit score
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=credit_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Credit Score"},
                gauge={
                    'axis': {'range': [300, 900], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [300, 500], 'color': "#FF5252"},  # Poor - Red
                        {'range': [500, 650], 'color': "#FFC107"},  # Average - Yellow
                        {'range': [650, 750], 'color': "#4CAF50"},  # Good - Green
                        {'range': [750, 900], 'color': "#1E88E5"}   # Excellent - Blue
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': credit_score
                    }
                }
            ))
            
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f'<p class="result-text">Rating: {rating}</p>', unsafe_allow_html=True)
            
        
        with col2:
            st.markdown(f'<p class="highlight">Default Probability</p>', unsafe_allow_html=True)
            
            # Create a pie chart for default probability
            fig = px.pie(
                values=[probability * 100, (1-probability) * 100],
                names=['Default Risk', 'Repayment Likelihood'],
                color_discrete_sequence=['#FF5252', '#4CAF50'],
                hole=0.6
            )
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            risk_level = "High" if probability > 0.5 else "Medium" if probability > 0.3 else "Low"
            st.markdown(f'<p class="result-text">Risk Level: {risk_level}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<p class="highlight">Recommendation</p>', unsafe_allow_html=True)
            
            # Generate recommendation based on credit score and default probability
            if credit_score >= 750:
                recommendation = "Low Risk"
                details = "Meets all criteria for standard approval process"
                icon = ""
            elif credit_score >= 650:
                recommendation = "Medium-Low Risk"
                details = "Standard approval with regular monitoring"
                icon = ""
            elif credit_score >= 500:
                recommendation = "Medium-High Risk"
                details = "Additional review required"
                icon = ""
            else:
                recommendation = "High Risk"
                details = "Recommend rejection or significant collateral"
                icon = ""
            
            st.markdown(f'<h1 style="text-align: center; font-size: 3rem;">{icon}</h1>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-text" style="text-align: center;">{recommendation}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align: center;">{details}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional insights section
        st.markdown('<div class="sub-header">Key Factors Affecting Your Score</div>', unsafe_allow_html=True)
        
        # Calculate loan to income ratio
        loan_to_income_ratio = st.session_state.input_data['loan_amount'] / st.session_state.input_data['income'] if st.session_state.input_data['income'] > 0 else 0
        
        # Create factors that affect the score
        factors = [
            {"factor": "Loan to Income Ratio", "impact": -0.8 if loan_to_income_ratio > 2 else -0.4 if loan_to_income_ratio > 1 else 0.2},
            {"factor": "Credit Utilization", "impact": -0.7 if st.session_state.input_data['credit_utilization_ratio'] > 70 else -0.3 if st.session_state.input_data['credit_utilization_ratio'] > 30 else 0.3},
            {"factor": "Delinquency History", "impact": -0.9 if st.session_state.input_data['delinquency_ratio'] > 50 else -0.5 if st.session_state.input_data['delinquency_ratio'] > 20 else 0.1},
            {"factor": "Residence Type", "impact": 0.4 if st.session_state.input_data['residence_type'] == "Owned" else -0.2 if st.session_state.input_data['residence_type'] == "Rented" else 0.1},
            {"factor": "Loan Type", "impact": -0.3 if st.session_state.input_data['loan_type'] == "Unsecured" else 0.3}
        ]
        
        # Convert to DataFrame for visualization
        df_factors = pd.DataFrame(factors)
        
        # Create horizontal bar chart
        fig = px.bar(
            df_factors, 
            x="impact", 
            y="factor", 
            orientation='h',
            color="impact", 
            color_continuous_scale=["#FF5252", "#FFFFFF", "#4CAF50"],
            range_color=[-1, 1],
            title="Factors Influencing Your Credit Assessment"
        )
        
        fig.update_layout(
            height=300,
            xaxis_title="Negative Impact ← → Positive Impact",
            yaxis_title="",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Improvement suggestions based on the weakest factors
        st.markdown('<p class="highlight">Suggestions for Improvement</p>', unsafe_allow_html=True)
        
        suggestions = []
        if loan_to_income_ratio > 2:
            suggestions.append("High loan-to-income ratio (>2.0): Consider additional collateral requirements.")
        if st.session_state.input_data['credit_utilization_ratio'] > 30:
            suggestions.append("High credit utilization (>30%): Flag for additional review.")
        if st.session_state.input_data['delinquency_ratio'] > 20:
            suggestions.append("Significant delinquency history (>20%): Recommend higher interest rate to offset risk.")
        if st.session_state.input_data['residence_type'] == "Rented":
            suggestions.append("Rented residence: Verify length of residence history and stability.")
        if st.session_state.input_data['loan_type'] == "Unsecured":
            suggestions.append("Unsecured loan: Consider offering secured alternatives if risk score is borderline.")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"**{i}.** {suggestion}")
        else:
            st.markdown("Your profile is already strong! Continue maintaining good financial habits.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please fill out the application form and click 'Calculate Credit Risk' to see results.")

# Help Tab
elif selected_tab == "ℹ️ Help":
    st.markdown('<div class="sub-header">Model Information & Documentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Risk Classification Matrix
    
    | Score Range | Risk Level | Internal Action |
    |------------|------------|------------------|
    | 750-900 | Low Risk | Standard approval process |
    | 650-749 | Medium-Low Risk | Standard approval with monitoring |
    | 500-649 | Medium-High Risk | Additional review required |
    | 300-499 | High Risk | Likely rejection or high collateral |
    
    ### Model Features
    
    - **Age**: Applicant's age in years
    - **Income**: Annual income in local currency
    - **Loan Amount**: Requested loan amount
    - **Loan Tenure**: Duration of loan in months
    - **Credit Utilization**: Percentage of available credit currently used
    - **Delinquency Ratio**: Percentage of loans with late payments
    - **Days Past Due**: Average number of days payments are late
    - **Residence Type**: Type of residence (Owned, Rented, Mortgage)
    - **Loan Purpose**: Purpose of the loan (Education, Home, Auto, Personal)
    - **Loan Type**: Whether secured or unsecured
    """)
    
    st.markdown('<div class="sub-header">Technical Documentation</div>', unsafe_allow_html=True)
    
    tech_docs = [
        {"section": "Model Training Process", 
         "content": "The model was trained on 37,500 historical loan records with an 8.6% default rate. Features were preprocessed using standard scaling and one-hot encoding for categorical variables. The final model uses logistic regression with L2 regularization (C=1.0)."},
        {"section": "Performance Metrics", 
         "content": "Accuracy: 92.4%, Precision: 89.2%, Recall: 76.5%, F1-Score: 82.4%, ROC AUC: 0.91. The model was validated using 5-fold cross-validation."},
        {"section": "Data Pipeline", 
         "content": "Input data is processed through a pipeline that handles missing values, scales numerical features, and encodes categorical variables. The prediction_helper.py file contains the implementation details."},
        {"section": "Model Limitations", 
         "content": "The model has limited performance on applicants with thin credit files (less than 1 year of credit history). For these cases, additional manual review is recommended."},
        {"section": "Deployment Information", 
         "content": "This application uses a serialized model stored in artifacts/model_data.joblib. Model was last updated on 2025-03-15 and should be retrained quarterly."}
    ]
    
    for i, doc in enumerate(tech_docs):
        with st.expander(doc["section"]):
            st.write(doc["content"])

# Footer
st.markdown('<div class="footer">© 2025 Internal Credit Risk Assessment Tool | Data Science Team</div>', unsafe_allow_html=True)
