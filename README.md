# UCLA Graduate Admission Predictor

This project uses machine learning to guess if a student might get into a Master's program at UCLA, based on their scores and profile. It includes a web app built with Streamlit where you can enter your details and get a prediction.

---

Visit the live Streamlit app here:
[https://uclaadmissionpredictionwithapp-4lcsl8gtld53ls7xnyrl4n.streamlit.app/](https://uclaadmissionpredictionwithapp-4lcsl8gtld53ls7xnyrl4n.streamlit.app/)

View the code on GitHub:
[https://github.com/MerwanCB/UCLA_Admission_Prediction_with_streamlit](https://github.com/MerwanCB/UCLA_Admission_Prediction_with_streamlit)

---

## Overview

This application predicts a prospective student's likelihood of admission into a Master's program at the University of California, Los Angeles (UCLA). Based on key academic and profile characteristics provided by the user, the model estimates whether the applicant has a 'High Chance' (>= 80% predicted probability) or 'Lower Chance' of being admitted. The goal is to provide students with a preliminary assessment to aid in their application process.

## Features

*   **User-Friendly Interface:** Powered by Streamlit for easy interaction.
*   **Input Form:** Collects details such as GRE/TOEFL scores, undergraduate CGPA, university rating, SOP/LOR strength, and research experience.
*   **Real-Time Prediction:** Instantly provides an admission chance prediction ('High Chance' / 'Lower Chance') based on the trained model.
*   **Accessible Online:** Deployed via Streamlit Community Cloud.

## Dataset

The model is trained on a dataset created specifically for predicting graduate admissions to UCLA. It includes the following parameters considered important during the application process:

*   **GRE_Score:** Graduate Record Examination score (out of 340).
*   **TOEFL_Score:** Test of English as a Foreign Language score (out of 120).
*   **University_Rating:** Rating of the undergraduate university (out of 5).
*   **SOP:** Statement of Purpose strength (out of 5).
*   **LOR:** Letter of Recommendation strength (out of 5).
*   **CGPA:** Undergraduate Cumulative Grade Point Average (out of 10).
*   **Research:** Research experience (Yes/No or 1/0).
*   **Admit_Chance:** The original target variable (probability), converted to a binary class (1 if >= 0.80, else 0) for this classification model.

## Technologies Used

*   **Streamlit:** For building and serving the interactive web application.
*   **Scikit-learn:** For the machine learning model (MLPClassifier), data splitting, preprocessing (MinMaxScaler), and evaluation metrics.
*   **Pandas:** For data loading, manipulation, and preparation.
*   **NumPy:** For numerical operations.
*   **Pickle:** For saving and loading the trained model and scaler.
*   **Matplotlib/Seaborn:** Used in the data analysis and model evaluation pipeline (e.g., generating loss curves).

## Model

The predictive core is a **Multi-Layer Perceptron (MLP) Neural Network**, a type of feedforward artificial neural network. It's trained as a classification model to predict the binary admission chance.

Key preprocessing steps applied before training include:
*   **One-Hot Encoding:** Converting categorical features (`University_Rating`, `Research`) into numerical format.
*   **Feature Scaling:** Using `MinMaxScaler` to scale numerical features to a common range (typically 0 to 1), which is often beneficial for neural network performance.

## Future Enhancements

*   Incorporate feature importance analysis to show users which factors most influence the prediction.
*   Experiment with different models (e.g., Gradient Boosting, SVM) to potentially improve accuracy.
*   Add more detailed explanations or feedback based on the prediction.
*   Allow users to upload a CSV file for batch predictions.
*   Integrate model explainability tools (like SHAP or LIME).

## Installation (for local deployment)

If you want to run the application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MerwanCB/UCLA_Admission_Prediction_with_streamlit.git
    cd UCLA_Admission_Prediction_with_streamlit
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app_streamlit.py
    ```

---

Thank you for checking out the UCLA Graduate Admission Predictor! Feedback is welcome.