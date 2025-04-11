def make_predictions(model, X_scaled):
    """
    Use a trained model to make predictions.
    """
    print("Making predictions...")
    # Generate predictions
    predictions = model.predict(X_scaled)
    print("Predictions complete.")
    return predictions