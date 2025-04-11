import os
import pickle

# --- Import Project Modules ---
from src.data.load_data import load_raw_data
from src.data.preprocess_data import (
    preprocess_target,
    drop_columns,
    change_dtype_to_object,
    save_processed_data,
)
from src.features.build_features import (
    create_dummies,
    split_features_target,
    split_train_test,
    scale_features,
)
from src.models.train_model import train_mlp_classifier
from src.models.predict_model import make_predictions
from src.models.evaluate_model import evaluate_classification
from src.visualization.visualize import plot_feature_scatter, plot_loss_curve

if __name__ == "__main__":

    # Define file paths and parameters
    RAW_DATA_FILE = "Admission.csv"
    PROCESSED_DATA_FILE = "admission_processed.csv"  
    TARGET_COLUMN = "Admit_Chance"
    THRESHOLD = 0.80
    COLUMNS_TO_DROP = ["Serial_No"]
    COLUMNS_TO_OBJECT = ["University_Rating", "Research"]
    COLUMNS_TO_DUMMY = ["University_Rating", "Research"]
    FIGURES_DIR = "reports/figures"
    SCATTER_PLOT_PATH = os.path.join(FIGURES_DIR, "gre_vs_toefl_scatter.png")
    LOSS_CURVE_PATH = os.path.join(FIGURES_DIR, "mlp_loss_curve.png")
    MODEL_SAVE_PATH = "models/mlp_model.pkl"
    SCALER_SAVE_PATH = "models/scaler.pkl"
    os.makedirs("models", exist_ok=True)  # Ensure models directory exists

    # Load raw data
    data = load_raw_data(RAW_DATA_FILE, data_dir="data/raw")

    if data is not None:
        # Preprocess data
        data_processed = preprocess_target(data, TARGET_COLUMN, THRESHOLD)
        data_processed = drop_columns(data_processed, COLUMNS_TO_DROP)
        data_processed = change_dtype_to_object(data_processed, COLUMNS_TO_OBJECT)
        save_processed_data(data_processed, PROCESSED_DATA_FILE, data_dir="data/processed")

        # Generate scatter plot for visualization
        plot_feature_scatter(
            data_processed.copy(),
            "GRE_Score",
            "TOEFL_Score",
            TARGET_COLUMN,
            SCATTER_PLOT_PATH,
        )

        # Create dummy variables for categorical features
        data_featured = create_dummies(data_processed, COLUMNS_TO_DUMMY)

        # Split data into features and target
        X, y = split_features_target(data_featured, TARGET_COLUMN)

        if X is not None and y is not None:
            # Split data into training and testing sets
            xtrain, xtest, ytrain, ytest = split_train_test(
                X, y, test_size=0.2, random_state=123, stratify=True
            )

            # Scale features
            scaler, xtrain_scaled, xtest_scaled = scale_features(xtrain, xtest)

            # Train MLP classifier
            model = train_mlp_classifier(
                xtrain_scaled,
                ytrain,
                hidden_layers=(3, 3),
                batch_size=50,
                max_iter=200,
                random_state=123,
            )

            # Save trained model
            try:
                with open(MODEL_SAVE_PATH, "wb") as f:
                    pickle.dump(model, f)
                print(f"Model saved successfully to {MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"Error saving model: {e}")

            # Save scaler
            try:
                with open(SCALER_SAVE_PATH, "wb") as f:
                    pickle.dump(scaler, f)
                print(f"Scaler saved successfully to {SCALER_SAVE_PATH}")
            except Exception as e:
                print(f"Error saving scaler: {e}")

            # Make predictions on test data
            ypred_test = make_predictions(model, xtest_scaled)

            # Evaluate model performance
            accuracy, confusion_mat = evaluate_classification(ytest, ypred_test)

            # Plot loss curve for the trained model
            plot_loss_curve(model, LOSS_CURVE_PATH)

            print("\n--- Workflow Complete ---")
        else:
            print("Exiting due to error in feature/target splitting.")
    else:
        print("Exiting due to data loading failure.")
