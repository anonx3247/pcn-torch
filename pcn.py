import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class PCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=torch.tanh, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.1)
        self.activation = activation
        self.device = device

    def pred_next(self, a):
        """
        Return pred_next from a and w
        """
        return self.activation(self.weight @ a)
    
    def pred_prev(self, a):
        """
        returns pred_prev from a and w
        """
        return self.activation(self.weight.T @ a)
    
    def err_next(self, a_next, a):
        """
        Returns e_next from a_next and a (-> pred_next)
        """
        return a_next - self.pred_next(a)
    
    def err_prev(self, a_prev, a):
        """
        Returns e_prev from a_prev and a (-> pred_prev)
        """
        return a_prev - self.pred_prev(a)

class PCN(nn.Module):
    def __init__(self, layer_sizes=None, layers=None, activation=torch.tanh, energy_function=F.mse_loss, device=None):
        super().__init__()
        assert layer_sizes is not None or layers is not None, "Either layer_sizes or layers must be provided"
        self.device = device if device is not None else torch.device('cpu')
        if layer_sizes is not None:
            self.layers = nn.ModuleList([
                PCNLayer(layer_sizes[i], layer_sizes[i+1], activation, device=self.device)
                for i in range(len(layer_sizes) - 1)
            ])
        elif layers is not None:
            self.layers = nn.ModuleList(layers)
        else:
            raise ValueError("Either layer_sizes or layers must be provided")
        
        self.energy_function = energy_function

    def backward_init(self, x, y):
        # Store activations for all layers as a list of vectors
        self.values = [None] * (len(self.layers) + 1) # 0 to L
        self.values[-1] = y.clone().to(self.device)  # output layer (target)
        
        # Initialize hidden layers with predictions
        for i in range(len(self.layers)-1, -1, -1):
            self.values[i] = self.layers[i].pred_prev(self.values[i+1]).detach().clone().to(self.device)
        self.values[0] = x.clone().to(self.device)  # input layer
        
    def backward_update(self, x, y, T=20, gamma=0.2, alpha=0.01, verbose=False):
        for t in range(T): # T steps in gradient descent
            # Compute errors for all layers

            if verbose:
                print('Computing errors...')
            
            errors = [0] * (len(self.layers) + 1)
            for i in range(len(self.layers)-1, -1, -1):
                a = self.values[i+1].detach()
                a_prev = self.values[i]
                errors[i] = self.layers[i].err_prev(a_prev, a)
            
            if verbose:
                print('Updating hidden activations...')
            
            for i in range(len(self.layers), 0, -1):
                a = self.values[i].detach()
                w_l = self.layers[i-1].weight.T
                error_l = errors[i]
                error_l_1 = errors[i-1] # okay because there is one more error than n_layers
                # Derivative of activation at next layer
                a_prev_pred = self.layers[i-1].pred_prev(a)
                if hasattr(self.layers[i-1].activation, 'derivative'):
                    act_deriv = self.layers[i-1].activation.derivative(a_prev_pred)
                else:
                    act_deriv = 1 - a_prev_pred ** 2  # assume tanh
                # Gradient: e^l - (W^{l})^T [e^{l+1} ∘ f'(W^{l} a^l)]
                grad = error_l - (w_l.T @ (error_l_1 * act_deriv))
                self.values[i] = (self.values[i] - gamma * grad).to(self.device)

        if verbose:
            print('Updating weights...')

        # Weight update step (after inference)
        for i in range(len(self.layers), 0, -1):
            a = self.values[i].detach()  # [in_features]
            w = self.layers[i-1].weight.T
            error_prev = errors[i-1]  # [out_features]
            preact = w @ a
            if hasattr(self.layers[i-1].activation, 'derivative'):
                act_deriv = self.layers[i-1].activation.derivative(preact)
            else:
                act_deriv = 1 - torch.tanh(preact) ** 2  # assume tanh
            # (error_next * act_deriv): [out_features]
            # a_prev: [in_features]
            # Outer product for weight update
            # delta_w = alpha * (error_next * act_deriv) @ a.T
            delta_w = alpha * torch.outer(a, error_prev * act_deriv)
            self.layers[i-1].weight.data += delta_w.to(self.device)


    def forward_init(self, x, y):
        # Store activations for all layers as a list of vectors
        self.values = [None] * (len(self.layers) + 1) # 0 to L
        self.values[0] = x.clone().to(self.device)  # input layer
        # Initialize hidden layers with predictions
        for i, layer in enumerate(self.layers[:-1]):
            self.values[i+1] = layer.pred_next(self.values[i]).detach().clone().to(self.device)
        
        self.values[-1] = y.clone().to(self.device)  # output layer (target)

    def forward_update(self, x, y, T=20, gamma=0.2, alpha=0.01, verbose=False):
        for t in range(T): # T steps in gradient descent
            # Compute errors for all layers

            if verbose:
                print('Computing errors...')

            errors = [0] # error with the input is 0
            for i, layer in enumerate(self.layers):
                a = self.values[i].detach()
                a_next = self.values[i+1] # okay because there is one more val in values than n_layers
                errors.append(layer.err_next(a_next, a))
            
            if verbose:
                print('Updating hidden activations...')


            # Update hidden activations (not input or output)
            for l, layer in enumerate(self.layers):
                a = self.values[l].detach()
                w_l = self.layers[l].weight
                error_l = errors[l]
                error_l_1 = errors[l+1] # okay because there is one more error than n_layers
                # Derivative of activation at next layer
                a_next_pred = self.layers[l].pred_next(a)
                if hasattr(self.layers[l].activation, 'derivative'):
                    act_deriv = self.layers[l].activation.derivative(a_next_pred)
                else:
                    act_deriv = 1 - a_next_pred ** 2  # assume tanh
                # Gradient: e^l - (W^{l})^T [e^{l+1} ∘ f'(W^{l} a^l)]
                grad = error_l - (w_l.T @ (error_l_1 * act_deriv))
                self.values[l] = (self.values[l] - gamma * grad).to(self.device)
        if verbose:
            print('Updating weights...')

        # Weight update step (after inference)
        for l, layer in enumerate(self.layers):
            a = self.values[l].detach()  # [in_features]
            w = layer.weight
            error_next = errors[l+1]  # [out_features]
            preact = w @ a
            if hasattr(layer.activation, 'derivative'):
                act_deriv = layer.activation.derivative(preact)
            else:
                act_deriv = 1 - torch.tanh(preact) ** 2  # assume tanh
            # (error_next * act_deriv): [out_features]
            # a_prev: [in_features]
            # Outer product for weight update
            # delta_w = alpha * (error_next * act_deriv) @ a.T
            delta_w = alpha * torch.outer(error_next * act_deriv, a)
            layer.weight.data += delta_w.to(self.device)

    def train_step(self, datapoint, T=20, gamma=0.2, alpha=0.01, verbose=False, forward=True):
        x, y = datapoint
        x = x.to(self.device)
        y = y.to(self.device)
        if forward:
            self.forward_init(x, y)
            self.forward_update(x, y, T, gamma, alpha, verbose=verbose)
        else:
            self.backward_init(x, y)
            self.backward_update(x, y, T, gamma, alpha, verbose=verbose)

    def train(self, train_loader, epochs=100, T=20, gamma=0.2, alpha=0.01, verbose=False, forward=True, evaluator=None):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for datapoint in tqdm(train_loader, desc="Training"):
                self.train_step(datapoint, T, gamma, alpha, verbose=verbose, forward=forward)
            if evaluator is not None:
                evaluator(self)
    
    def classify(self, x):
        # Store activations for all layers as a list of vectors
        self.values = [None] * (len(self.layers) + 1)
        self.values[0] = x.clone().to(self.device)  # input layer
        # Initialize hidden layers with predictions
        for i, layer in enumerate(self.layers):
            self.values[i+1] = layer.pred_next(self.values[i]).detach().clone().to(self.device)
        return self.values[-1]
    
    def generate(self, y):
        # we move backwards from the output layer to the input layer
        # Store activations for all layers as a list of vectors
        self.values = [None] * (len(self.layers) + 1)
        self.values[-1] = y.clone().to(self.device)  # output layer
        # Initialize hidden layers with predictions
        for i in range(len(self.layers)-1, -1, -1):
            self.values[i] = self.layers[i].pred_prev(self.values[i+1]).detach().clone().to(self.device)
        return self.values[0]
    
    def evaluate(self, test_loader):
        acc = 0
        for x, y in test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.classify(x)
            y_hat = torch.argmax(y_hat)
            y = torch.argmax(y)
            if y_hat == y:
                acc += 1
        return acc / len(test_loader)

