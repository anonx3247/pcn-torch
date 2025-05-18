# Predictive Coding Network (PCN) in PyTorch

This repository provides a PyTorch implementation of a Predictive Coding Network (PCN), a biologically inspired neural network model that learns by minimizing prediction errors across layers. The code is designed for flexibility and educational clarity, supporting both forward and backward predictive coding schemes.

## Features
- Modular PCN layers with customizable activation functions
- Forward and backward predictive coding algorithms
- Weight and activation updates based on local errors
- Training and evaluation loops compatible with PyTorch DataLoaders
- Classification and generative modes

## What is a PCN and how is it different from a regular Neural Network?

A Predictive Coding Network (PCN) differs from traditional neural networks in several key ways:

1. **Error-Driven Learning**: While traditional neural networks use backpropagation to update weights based on global error signals, PCNs learn by minimizing local prediction errors at each layer. Each layer tries to predict the activity of the next layer, and learning occurs by adjusting weights to reduce these prediction errors.

2. **Bidirectional Information Flow**: PCNs maintain both feedforward and feedback connections between layers. The feedforward connections carry predictions, while feedback connections carry error signals. This creates a continuous cycle of prediction and error correction.

3. **Local Learning Rules**: In PCNs, each layer updates its weights based only on local information - the prediction errors it receives and the activities of connected layers. This makes PCNs more biologically plausible than traditional neural networks, which require global error signals.

4. **Dynamic Inference**: During inference, PCNs don't just pass information forward once. Instead, they engage in an iterative process where activations are updated multiple times to minimize prediction errors across the network. This can be done in either a forward or backward manner:
   - Forward PCN: Updates propagate from input to output
   - Backward PCN: Updates propagate from output to input

5. **Generative Capabilities**: Due to their bidirectional nature, PCNs can not only classify inputs but also generate data by starting from an output pattern and propagating backwards through the network.

These differences make PCNs particularly interesting for both neuroscience research and machine learning applications, as they offer a more biologically plausible approach to learning while maintaining competitive performance on various tasks.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd pcn-torch
   ```
2. Install dependencies (requires Python 3.8+):
   ```bash
   pip install torch tqdm
   ```

## Usage

### Basic Example
```python
import torch
from pcn import PCN

# Define network architecture (e.g., input-100-10 for MNIST)
layer_sizes = [784, 100, 10]
model = PCN(layer_sizes=layer_sizes, device='cpu')

# Example data (replace with real DataLoader for practical use)
x = torch.randn(784)  # Example input
y = torch.zeros(10); y[3] = 1  # Example one-hot target

# Single training step
model.train_step((x, y), T=20, gamma=0.2, alpha=0.01, forward=True)

# Classification
output = model.classify(x)
predicted_class = torch.argmax(output)
```

### Training with DataLoader
```python
from torch.utils.data import DataLoader, TensorDataset

# Prepare your dataset
X = torch.randn(100, 784)
Y = torch.nn.functional.one_hot(torch.randint(0, 10, (100,)), num_classes=10).float()
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Train the model
model.train(loader, epochs=10, T=20, gamma=0.2, alpha=0.01, forward=True)
```

## API

### PCNLayer
- `PCNLayer(in_features, out_features, activation=torch.tanh, device=None)`
  - Basic building block for PCN. Handles predictions and error computation for a single layer.

### PCN
- `PCN(layer_sizes=None, layers=None, activation=torch.tanh, energy_function=F.mse_loss, device=None)`
  - Main network class. Accepts a list of layer sizes or custom PCNLayer instances.
- `train_step(datapoint, T=20, gamma=0.2, alpha=0.01, verbose=False, forward=True)`
  - Performs a single PCN update step on a data point.
- `train(train_loader, epochs=100, T=20, gamma=0.2, alpha=0.01, verbose=False, forward=True, evaluator=None)`
  - Trains the network over multiple epochs using a DataLoader.
- `classify(x)`
  - Runs the network in classification mode (forward pass).
- `generate(y)`
  - Runs the network in generative mode (backward pass).
- `evaluate(test_loader)`
  - Evaluates classification accuracy on a test DataLoader.

## Parameters
- `T`: Number of inference steps per sample
- `gamma`: Step size for activation updates
- `alpha`: Learning rate for weight updates
- `forward`: If True, uses forward predictive coding; if False, uses backward

## References
- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience, 2(1), 79-87.
- Whittington, J. C., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural computation, 29(5), 1229-1262.

## License
MIT License
