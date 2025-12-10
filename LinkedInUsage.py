import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    try:
        s = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded.")

        # Display the first few rows of the data
        st.write("Data Preview:")
        st.dataframe(s.head())

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    st.info("Please upload a CSV file to begin.")

# CODE FROM NOTEBOOK TO PREPARE THE MODEL

# Read in file, call it "s"
#s = pd.read_csv("social_media_usage.csv")

# Define function clean_sm, if x=1, return 1, else 0
def clean_sm(x):
    # Check to see if x is a dataframe (originally just had the np.where returned but the resultant array was not a dataframe and it looked funny)
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(np.where(x == 1, 1, 0))
    else:
        return np.where(x == 1, 1, 0)

# Create SS
data = {
    'sm_li' : clean_sm(s['web1h']),
    'income' : s['income'],
    'education' : s['educ2'],
    'parent' : s['par'],
    'married' : s['marital'],
    'female' : s['gender'],
    'age' : s['age']
                       }

ss = pd.DataFrame(data)

# Clean the Dataframe
ss.parent = clean_sm(ss.parent)
ss.married = clean_sm(ss.married)
ss.female = clean_sm(ss.female)

# Use np.where instead of the above commented out code
ss.income = np.where(ss.income > 9, np.nan, ss.income)
ss.education = np.where(ss.education > 8, np.nan, ss.education)
ss.age = np.where(ss.age > 98, np.nan, ss.age)

# Drop any missing values
ss = ss.dropna()

# Define target and features
target = ss['sm_li']
features = ss.drop(columns=['sm_li'])

# Split the data into training and test sets. Hold out 20% of the data for testing.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Instantiate a logistic regression model with class_weight set to balanced
model = LogisticRegression(class_weight='balanced', random_state=42)

# Fit the model with the training data
model.fit(X_train, y_train)

# STREAMLIT APP CODE BELOW
st.markdown('<div style="text-align: right;">OPAN6607 Final Project</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: right;">Rick Mastropole</div>', unsafe_allow_html=True)

st.title("LinkedIn User Prediction App")
st.write("This app predicts whether an individual is a LinkedIn user based on their demographic information.")


# Get user input
with st.container(border=True):
    st.write("Please input the following demographic information:")
    age = st.slider("**Age**", 18, 98, 35)
    income = st.slider("**Income** (1-9)", 1, 9, 5)
    education = st.slider("**Education Level** (1-8)", 1, 8, 4)

# Put the Binary inputs in columns since it looks cleaner
    col1, col2, col3 = st.columns(3)
    with col1:
        parent = st.selectbox("**Parent** (0 = No, 1 = Yes)", [0, 1])
    with col2:
        married = st.selectbox("**Married** (0 = No, 1 = Yes)", [0, 1])
    with col3:
        female = st.selectbox("**Female** (0 = No, 1 = Yes)", [0, 1])

# Create a DataFrame for the input
input = pd.DataFrame({
    'income': [income],
    'education': [education],
    'parent': [parent],
    'married': [married],
    'female': [female],
    'age': [age]    
})

# Make prediction
pred = model.predict(input)[0]
prob = model.predict_proba(input)[0][1]

# Display the result
with st.container(border=True):
    st.write(f"**Predicted LinkedIn User:** {'Yes' if pred == 1 else 'No'}")
    st.write(f"**Probability of being a LinkedIn User:** {round(prob,2)}")

with st.container(border=True):
    # Visualizations for Marketing Team
    # Feature Coefficients
    coefficients = pd.Series(model.coef_[0], index=features.columns).sort_values(ascending=False)

    # Predicted Probabilities for the test features
    test_prob = model.predict_proba(X_test)[:, 1]

    st.header("Marketing Insights")
    st.markdown("Helpful visuals to demonstrate overall model performance and the relative impact of each feature. Use these to help guide effective marketing strategies.")

    # Coefficient Plot
    with st.expander("Feature Importance", expanded=True):
        st.subheader("Which features are most influential to LinkedIn usage?")
        fig_coef, ax_coef = plt.subplots(figsize=(8, 5))
        sns.barplot(x=coefficients.values, y=coefficients.index, palette="colorblind", ax=ax_coef)
        ax_coef.set_title("Logistic Regression Model Coefficients")
        ax_coef.set_xlabel("Coefficient Value (Impact on Log-Odds)")
        ax_coef.set_ylabel("Demographic Feature")
        plt.tight_layout()
        st.pyplot(fig_coef)
        st.caption("**Insight:** Positive values significantly increase the probability of being a LinkedIn user, making these ideal targeting factors. From the coefficients, education and income have the highest positive impact, while age has a negative impact.")

    # Probability Distribution Plot
    with st.expander("Predicted Probability Distribution", expanded=True):
        st.subheader("Marketing Target Confidence Across Test Population")
        fig_prob, ax_prob = plt.subplots(figsize=(8, 5))
        sns.histplot(test_prob, bins=30, kde=True, ax=ax_prob, color="#055999")
        ax_prob.set_title("Distribution of Predicted LinkedIn User Probabilities")
        ax_prob.set_xlabel("Predicted Probability of LinkedIn Use")
        ax_prob.set_ylabel("Count of Individuals")
        plt.tight_layout()
        st.pyplot(fig_prob)
        st.caption("**Insight:** Individuals with probabilities clustered near 1.0 are the most reliable marketing targets. Those near 0.5 are the most uncertain with those near 0.0 as almost certainly not marketing targets.")