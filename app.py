import streamlit as st
import numpy as np
import xgboost as xgb
import pickle

def main():
    st.title('Credit Approval Prediction')
    st.write('This app predicts credit approval based on user input.')

    model_file = "gmsc3.pkl"
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    # Input fields
    # SeriousDlqin2yrs = st.slider('Dalam 2 tahun berapa kali menunggak 30 - 59 hari', 0, 20)
    RevolvingUtilizationOfUnsecuredLines = st.number_input('Total saldo kartu kredit', 0.0)
    age = st.slider('Umur', 18, 110)
    NumberOfTime30_59DaysPastDueNotWorse = st.slider('Jumlah menunggak dalam durasi 30 - 58 hari', 0, 10)
    DebtRatio = st.number_input('Ratio Hutang', 0.0, 1.0)
    MonthlyIncome = st.number_input('Pendapatan Perbulan')
    NumberOfOpenCreditLinesAndLoans = st.slider('Jumlah Cicilan ', 0, 20)
    NumberOfTimes90DaysLate = st.slider('Jumlah menunggak lebih dari 90 hari', 0, 20)
    NumberRealEstateLoansOrLines = st.slider('Jumlah cicilan property', 0, 20)
    NumberOfTime60_89DaysPastDueNotWorse = st.slider('Jumlah menunggak dalam durasi 60 - 89 hari', 0, 20)
    NumberOfDependents = st.slider('Jumlah tanggungan', 0, 10)

    user_input = np.array([[RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30_59DaysPastDueNotWorse,DebtRatio,MonthlyIncome, NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,NumberRealEstateLoansOrLines,NumberOfTime60_89DaysPastDueNotWorse,NumberOfDependents]])  # Adjust based on your model's input requirements

    if st.button('Predict Approval'):
        user_input = np.array([[RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30_59DaysPastDueNotWorse, DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines, NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents]])
        prediction = model.predict(user_input)
        prediction_probability = model.predict_proba(user_input)[0][1]

        st.write('Prediction:', prediction[0])
        st.write('Approval Probability:', prediction_probability)

if __name__ == '__main__':
    main()
