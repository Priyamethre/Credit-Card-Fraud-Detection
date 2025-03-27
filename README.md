# Credit Card Fraud Detection System  

## **Overview**  
Credit card fraud is a major financial risk for banks, businesses, and customers, leading to billions of dollars in losses annually. Traditional fraud detection methods rely on rule-based systems, which often fail to adapt to evolving fraud patterns. This project leverages **Machine Learning (ML)** to build an automated fraud detection system that can identify fraudulent transactions with high accuracy while minimizing false positives.

## **Business Objective**  
 * Detect fraudulent transactions to prevent unauthorized payments.  
 * Minimize financial losses for banks and financial institutions.  
* Reduce false positives, ensuring legitimate transactions are not mistakenly flagged.  
 * Ensure scalability, making the system efficient for handling large transaction volumes.  

## **Dataset**  
- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Size:** 284,807 transactions  
- **Features:**  
  - **Time & Amount:** Transaction time and transaction amount.  
  - **V1 to V28:** Features obtained from **PCA transformation**.  
  - **Class:** Target variable (**0 = Non-Fraud, 1 = Fraud**).  
- **Challenge:** Highly imbalanced data (**Only 0.17% fraud cases**).  

## **Exploratory Data Analysis (EDA)**  
Performed in-depth **EDA** using **Matplotlib, Seaborn, and SweetViz**, including:  
 * Class distribution analysis (Fraud vs. Non-Fraud)  
 * Correlation heatmaps to understand relationships between features  
 * Transaction amount distribution analysis  
* PCA scatter plots for feature visualization  
* Fraudulent transaction time analysis 

## **Model Development**  
* The following **Machine Learning models** were trained and evaluated:  
 **Random Forest**  
**AdaBoost**  
 **Decision Tree**  
 **XGBoost (Best Performing Model)**  

* **Handling Class Imbalance:**  
Since the dataset was highly imbalanced, **SMOTE (Synthetic Minority Over-sampling Technique)** was used to balance the dataset before training.  

## **Best Model - XGBoost**  
 * **AUC Score:** **0.9776** (on test data)  
*  **Feature Importance:** Key features - **V14, V4, V17, V12, V10**  
*  **Handles class imbalance** better than traditional models  
*  **High precision and recall**, reducing false positives and false negatives  

## **Deployment - Streamlit Web App**  
An interactive **Streamlit Web App** was developed for fraud detection, allowing users to:  
 **Upload transaction data**  
 **Get real-time fraud predictions**  
**View fraud probability scores**  


# Conclusion
*  XGBoost was the best-performing model, achieving a validation AUC of 0.984 and test AUC of 0.9776, proving it generalizes well.
*  Decision Tree (AUC: 0.9369) performed slightly better than Random Forest and AdaBoost but was not as strong as XGBoost.
*  SMOTE was used to balance the dataset, improving fraud detection without overfitting.
*  A Streamlit-based web app was built, allowing users to upload transactions and get fraud predictions instantly.
*  This solution provides an efficient, scalable, and accurate fraud detection system that can be further improved with deep learning and real-time deployment on cloud platforms.
