from sklearn.neural_network import MLPClassifier

def train_mlp_classifier(xtrain_scaled, ytrain, hidden_layers=(3,3), batch_size=50, max_iter=200, random_state=123):
    """
    Train an MLPClassifier model with specified parameters.
    """
    print(f"Training MLPClassifier with parameters: hidden_layers={hidden_layers}, batch_size={batch_size}, max_iter={max_iter}, random_state={random_state}")
    # Initialize the MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state
    )
    # Train the model
    mlp.fit(xtrain_scaled, ytrain)
    print("MLPClassifier training complete.")
    return mlp