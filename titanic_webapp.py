# -*- coding: utf-8 -*-

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ai_wonder as wonder

# The loader
@st.cache_resource
def load_context(dataname):
    state = wonder.load_state(f"{dataname}_state.pkl")
    model = wonder.input_piped_model(state)
    return state, model

# The driver
if __name__ == "__main__":
    # Streamlit interface
    st.subheader(f"Titanic 'Survived' Predictor")
    st.markdown("Powered by :blue[**AI Wonder**]")
    st.markdown("")

    # Arrange radio buttons horizontally
    st.write('<style> div.row-widget.stRadio > div { flex-direction: row; } </style>',
        unsafe_allow_html=True)

    # User inputs
    Sex = st.radio("Sex", ['male', 'female'], index=0)
    Embarked = st.radio("Embarked", ['S', 'C', 'Q'], index=2)
    Pclass = st.number_input("Pclass", value=1)
    Age = st.number_input("Age", value=34.0)
    SibSp = st.number_input("SibSp", value=0)
    Parch = st.number_input("Parch", value=0)
    Fare = st.number_input("Fare", value=26.55)

    st.markdown("")

    # Make datapoint from user input
    point = pd.DataFrame([{
        'Sex': Sex,
        'Embarked': Embarked,
        'Pclass': Pclass,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
    }])

    # Load context
    state, model = load_context('titanic')

    # Predict and Explain
    if st.button('Predict'):
        st.markdown("")

        with st.spinner("Making predictions..."):
            prediction = str(model.predict(point)[0])
            st.success(f"Prediction of **{state.target}** is **{prediction}**.")
            predprobas = zip(['0', '1'],
                np.round(model.predict_proba(point)[0], 2))
            predprobas_str = ", ".join([f"{label}: {proba}" for label, proba in predprobas])
            st.success(f"Probabilities are **{predprobas_str}**.")
            st.markdown("")

        with st.spinner("Making explanations..."):
            st.info("Feature Importances")
            importances = pd.DataFrame(wonder.local_explanations(state, point), columns=["Feature", "Value", "Importance"])
            st.dataframe(importances.round(2))

            st.info("Some Counterfactuals")
            counterfactuals = wonder.whatif_instances(state, point).iloc[:20]
            st.dataframe(counterfactuals.round(2))
