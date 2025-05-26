import matplotlib.pyplot as plt
from FFNN import FFNN

def plot_training_comparison(training_history):
    plt.figure(figsize=(10, 6))
    plt.plot(
        training_history['training_loss_history'], 
        label='Loss on Training Data', 
        color='blue'
    )
    plt.plot(
        training_history['validation_loss_history'], 
        label='Loss on Validation Data', 
        color='red', 
        linestyle='--'
    )
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_weight_distribution(ffnn: FFNN):
    num_layers = len(ffnn.layers)
    
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 4 * num_layers))
    
    if num_layers == 1:
        axes = [axes]

    for i, layer in enumerate(ffnn.layers):
        weights = layer.W.data.flatten()
        gradients = layer.W.grad.flatten() if layer.W.grad is not None else None

        axes[i][0].hist(weights, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[i][0].set_title(f"Layer {i+1} - Weight Distribution")
        axes[i][0].set_xlabel("Weight Value")
        axes[i][0].set_ylabel("Frequency")

        if gradients is not None:
            axes[i][1].hist(gradients, bins=30, color='red', alpha=0.7, edgecolor='black')
            axes[i][1].set_title(f"Layer {i+1} - Gradient Distribution")
            axes[i][1].set_xlabel("Gradient Value")
            axes[i][1].set_ylabel("Frequency")
        else:
            axes[i][1].axis("off")
    
    plt.tight_layout()
    plt.show()