import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Function to load data
@st.cache_resource
def load_data():
    return pd.read_csv('tips.csv')  # Replace with your dataset path

# Load the dataset
tips = load_data().copy()

# Advanced CSS
advanced_css = """
<style>
body {
    font-family: 'Times New Roman', serif;
    background-color: #F5F5F5;
    margin:10px;
}
.custom-header {
    font-size: 26px;
    font-weight: bold;
    color: #4A4A4A;
    padding-bottom: 10px;
    margin-top: 30px;
    border-bottom: 2px solid #FF4B4B;
    margin-bottom: 20px;
}



h1 
{font-size: 30px;
        font-weight: bold;
        text-align: center;
        background-color: #3380FF;
        width=703px;
        padding: 10px;
        opacity: 0.8;
        color: #FFFFFF;
         margin-bottom: 20px;
        }

 .st-bq {
    color: #0B5ED7;
    text-align: center;
}



h2,h3,.stButton>button {
    color: #FFFFFF;
    background-color: #3380FF;
    padding: 5px;
    #border-radius: 5px;
    text-align: center;
    padding: 10px;
    opacity: 0.6;
    
  
}

.stTextInput>div>div>input, .stNumberInput>div>div>input {
    border-radius: 5px;
   
}

.st-bx {
    background-color: white;
    margin-bottom: 20px;
}

.sidebar .sidebar-content {
    background-color: #0B5ED7;
    color: white;
   
}

.stButton>button {
    width: 100%;
    border-radius: 5px;
    border: 1px solid #0B5ED7;
   
}

.stButton>button:hover {
    background-color: #0056b3;
}

.stPlotlyChart {
    border-radius: 10px;
}
</style>
"""

# Inject Advanced CSS
st.markdown(advanced_css, unsafe_allow_html=True)

# Title and Description
st.title('Interactive Tip Prediction')
st.write('Predict the tip amount based on your total bill and explore tipping trends.')

with st.sidebar:
    st.header('Predict Your Tip')

    # Display the image in the sidebar
    st.image('tip.jpeg', use_column_width=True)

    total_bill = st.number_input('Enter Total Bill Amount', min_value=0.0, format="%.2f")
    submit_button = st.button('Predict Tip')

# Prepare the data for SLR
X = tips[['total_bill']]
Y = tips['tip']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit the simple linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Prediction and Displaying the Result
if submit_button:
    if total_bill > 0:
        predicted_tip = model.predict([[total_bill]])[0]
        st.success(f'Predicted Tip: ${predicted_tip:.2f}')
    else:
        st.error('Please enter a valid total bill amount.')

# Interactive Section
# Custom styled header with markdown
st.markdown('<div class="custom-header">Explore Tipping Trends</div>', unsafe_allow_html=True)
selected_metric = st.selectbox('Choose a metric to explore:', 
                               ['Tip as a Percentage of Total Bill', 'Average Tip by Bill Range'], 
                               key='select_metric')  # Unique key for this selectbox

if selected_metric == 'Tip as a Percentage of Total Bill':
    tips['tip_percentage'] = (tips['tip'] / tips['total_bill']) * 100
    fig = px.histogram(tips, x='tip_percentage', nbins=20, title='Tip Percentage Distribution', labels={'tip_percentage': 'Tip as Percentage of Total Bill'})
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig, use_container_width=True)

elif selected_metric == 'Average Tip by Bill Range':
    tips['bill_range'] = pd.cut(tips['total_bill'], bins=np.arange(0, 60, 10))
    # Convert Interval objects to strings for Plotly
    tips['bill_range_str'] = tips['bill_range'].astype(str)
    avg_tips = tips.groupby('bill_range_str')['tip'].mean().reset_index()
    
    fig = px.bar(avg_tips, x='bill_range_str', y='tip', title='Average Tip by Bill Range', 
                 labels={'bill_range_str': 'Total Bill Range ($)', 'tip': 'Average Tip ($)'})
    fig.update_traces(marker_color='pink')
    st.plotly_chart(fig, use_container_width=True)



