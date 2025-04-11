import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_scatter(df, x_col, y_col, hue_col, save_path="reports/figures/feature_scatter.png"):
    """
    Create and save a scatter plot of two features, colored by a third column.
    """
    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
    plt.title(f'Scatter Plot: {x_col} vs {y_col} (Colored by {hue_col})')

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
    print(f"Scatter plot saved to {save_path}")

def plot_loss_curve(model, save_path="reports/figures/loss_curve.png"):
    """
    Plot and save the loss curve from a trained MLP model.
    """
    if hasattr(model, 'loss_curve_'):
        # Extract loss values and plot
        loss_values = model.loss_curve_
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Loss', color='blue')
        plt.title('MLP Training Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Ensure the save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()  # Close the plot to free memory
        print(f"Loss curve plot saved to {save_path}")
    else:
        print("Warning: Model does not have 'loss_curve_' attribute. Cannot plot loss curve.")