**How to Run**

To reproduce our analysis, visualizations, and generate the check-in report, run the following script from the root of this directory:
```bash
python prepare_checkin.py
```
This requires `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `ucimlrepo` to be installed.

**Topic**

This project studies hospital readmissions among diabetic patients. Hospital readmissions within 30 days are costly and are used as a quality metric in healthcare systems. Using clinical and demographic data, we aim to analyze patterns associated with readmission risk.

**Current Interesting Health Markers**

1. Age
2. Previous Admissions
3. Number of procedures
4. Number of current medications
5. Diabetes medication
6. Time in hospital

**Timeline**
1. Decide which features and data points to use ( Week 1 )
2. Data Cleaning and Preproccessing( Week 2 )
    - Remove missing/invalid values
    - Split dataset into training and testing sets
3. Select and Develop Initial Model ( Week 3-5 )
    - Build Basic Model (Unsure as of now)
    - Decide which tests to impliment
4. Model Evaluation ( Week 5-7 )
    - Evaluate model peformance on test data
    - Adjust model depending on results
    - Ensure accurate predictions by accuracy
    - Analyze top characteristics that increase readmission risk
6. Create Risk Groups ( Week 8 )
    - Depending on probabilities create low, medium and high risk groups
7. Create Visulizations and Presentation ( Week 9-11 )
    - Create presentation findings
       - Important Predictors
       - Importance of this data
       - Risk Groups
       - ...
    - Dicussion of importance
    - Create visual of model  


**Goals**

1. Successfully predict if a diabetic patient will be readmitted within 30 days based on clinical or demographic features. 
2. Identify the 3 most influential predictors of readdmission.
3. Develop a paitent risk profile of if they will return. (Maybe risk groups)

**Potential Data sources identified**

Main Dataset -
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
https://www.kaggle.com/datasets/brandao/diabetes/data

Possible test dataset -  https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset (Synthetic Data)

**Data collection method explained**

We will obtain the dataset from publicly available sources, primarily Kaggle and the UCI Machine Learning Repository. The data can be directly downloaded as CSV files and imported for preprocessing and analysis.

**Test Plan**

The dataset will be divided into seperate training and testing sections. 50% will train the model and 50% will be used for testing the model.
