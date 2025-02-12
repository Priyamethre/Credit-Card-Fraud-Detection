# -*- coding: utf-8 -*-
"""
streamlit demo for real time
"""
import streamlit as st

st.set_page_config(page_title="Fraud Detection App", layout="wide")

st.title("💳 Credit Card Fraud Detection System")

st.sidebar.success("Select a page above to proceed.")


# Use st.markdown with HTML for styling
st.markdown(
    """
    <style>
    .green-title {
        color:green;
        font-size: 28px;
        font-weight: bold;
    }
    .content-text {
        font-size: 20px;
        text-align: justify;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Overview Section
st.markdown('<p class="green-title">Overview:</p>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="content-text">
    With the increase of people using credit cards in their daily lives, credit card companies should take special care of the security and safety of the customers. According to (Credit card statistics 2021), the number of people using credit cards worldwide was 2.8 billion in 2019; also, 70those users own a single card. Reports of Credit card fraud in the U.S. rose by 44.7in 2020. There are two kinds of credit card fraud, and the first is having a credit card account opened under your name by an identity thief. Reports of this fraudulent behaviour increased 48to 2020. The second type is when an identity thief uses an existing account you created, usually by stealing the information on the credit card. Reports on this type of Fraud increased 9to 2020(Daly, 2021). Those statistics caught We’s attention as the numbers have increased drastically and rapidly throughout the years, which motivated We to resolve the issue analytically by using different machine learning methods to detect fraudulent credit card transactions within numerous transactions.
    </p>
    """,
    unsafe_allow_html=True
)

# Business Objective Section
st.markdown('<p class="green-title">Business Objective:</p>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="content-text">
    The main aim of this project is the detection of fraudulent credit card transactions, as it is essential to figure out the fraudulent transactions so that customers do not get charged for the purchase of products that they did not buy. Fraudulent Credit card transactions will be detected with multiple ML techniques. Then, a comparison will be made between the outcomes and results of each method to find the best and most suited model for detecting fraudulent credit card transactions; graphs and numbers will also be provided. In addition, it explores previous literature and different techniques used to distinguish Fraud within a dataset.
    </p>
    """,
    unsafe_allow_html=True
)


st.image(r"C:\Users\priya\OneDrive\Desktop\PRIYA\internship at 360 digitmg\CREDIT CARD\credit.jpg",width=1200)
























