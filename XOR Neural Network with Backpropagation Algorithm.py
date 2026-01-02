# -*- coding: utf-8 -*-
"""
XOR Neural Network with Backpropagation Algorithm
Network Architecture: 2 input nodes -> 2 hidden neurons -> 1 output neuron
"""

import numpy as np

class XORNeuralNetwork:
    def __init__(self):
        """Initialize the neural network with XOR data"""
        # XOR gate training data
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input patterns
        self.y = np.array([[0], [1], [1], [0]])              # Expected outputs
        
        # Initialize weights and biases with random values
        np.random.seed(42)  # For reproducible results
        
        # Hidden layer weights: 2 inputs -> 2 hidden neurons
        # W1 shape: (2, 2) - connects 2 inputs to 2 hidden neurons
        self.W1 = np.random.randn(2, 2) * 0.1  # Small random values
        self.b1 = np.zeros((1, 2))  # Zero initialization for biases
        
        # Output layer weights: 2 hidden neurons -> 1 output
        # W2 shape: (2, 1) - connects 2 hidden neurons to 1 output
        self.W2 = np.random.randn(2, 1) * 0.1  # Small random values
        self.b2 = np.zeros((1, 1))  # Zero initialization for bias
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation: compute network output from input"""
        # Input to hidden layer
        # z1 = X·W1 + b1 (weighted sum for hidden layer)
        self.z1 = np.dot(X, self.W1) + self.b1
        # a1 = σ(z1) (hidden layer activation)
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden layer to output
        # z2 = a1·W2 + b2 (weighted sum for output layer)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # a2 = σ(z2) (output layer activation)
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        """Backward propagation: compute gradients and update weights"""
        m = X.shape[0]  # Number of training examples
        
        # Calculate error at output layer
        # δ_output = (a2 - y) * σ'(a2)
        output_error = self.a2 - y
        d_output = output_error * self.sigmoid_derivative(self.a2)
        
        # Calculate gradient for output layer weights and biases
        # ∂Loss/∂W2 = (1/m) * a1ᵀ · δ_output
        # ∂Loss/∂b2 = (1/m) * Σ δ_output
        dW2 = (1/m) * np.dot(self.a1.T, d_output)
        db2 = (1/m) * np.sum(d_output, axis=0, keepdims=True)
        
        # Calculate error at hidden layer
        # δ_hidden = (δ_output · W2ᵀ) * σ'(a1)
        hidden_error = np.dot(d_output, self.W2.T)
        d_hidden = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Calculate gradient for hidden layer weights and biases
        # ∂Loss/∂W1 = (1/m) * Xᵀ · δ_hidden
        # ∂Loss/∂b1 = (1/m) * Σ δ_hidden
        dW1 = (1/m) * np.dot(X.T, d_hidden)
        db1 = (1/m) * np.sum(d_hidden, axis=0, keepdims=True)
        
        # Update weights and biases using gradient descent
        # W = W - α * ∂Loss/∂W
        # b = b - α * ∂Loss/∂b
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, epochs=10000, learning_rate=0.1, verbose=True):
        """Train the neural network"""
        print("Starting training...")
        print(f"Network Architecture: 2 inputs -> 2 hidden neurons -> 1 output")
        print(f"Training epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print("-" * 50)
        
        losses = []  # To track loss history
        for epoch in range(epochs):
            # Forward pass
            outputs = self.forward(self.X)
            
            # Calculate mean squared error
            loss = np.mean((outputs - self.y) ** 2)
            losses.append(loss)
            
            # Backward pass and weight update
            self.backward(self.X, self.y, learning_rate)
            
            # Print progress every 1000 epochs
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        print("-" * 50)
        print(f"Final loss: {loss:.6f}")
        print("Training completed!")
        return losses
    
    def predict(self, inputs):
        """Make prediction using trained network"""
        # Ensure inputs are in correct shape (2D array)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        
        # Forward pass to get prediction
        prediction = self.forward(inputs)
        
        # Convert to binary output (0 or 1)
        binary_prediction = 1 if prediction[0, 0] > 0.5 else 0
        
        return binary_prediction, prediction[0, 0]
    
    def test(self):
        """Test the network on all XOR cases"""
        print("\n" + "=" * 50)
        print("XOR Gate Testing")
        print("=" * 50)
        
        # Test all 4 XOR combinations
        test_results = []
        for i in range(len(self.X)):
            input_val = self.X[i]
            expected = self.y[i, 0]
            binary_pred, prob = self.predict(input_val)
            
            result_str = f"Input: {input_val} -> Expected: {expected}, Predicted: {binary_pred} (Probability: {prob:.4f})"
            test_results.append((expected, binary_pred, prob))
            print(result_str)
        
        # Calculate accuracy
        correct = sum(1 for expected, pred, _ in test_results if expected == pred)
        accuracy = correct / len(test_results) * 100
        
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(test_results)} correct)")
        return accuracy, test_results
    
    def display_network_info(self):
        """Display network architecture and weights"""
        print("\n" + "=" * 50)
        print("Neural Network Information")
        print("=" * 50)
        print("Network Architecture:")
        print("  Input Layer: 2 nodes")
        print("  Hidden Layer: 2 neurons (with sigmoid activation)")
        print("  Output Layer: 1 neuron (with sigmoid activation)")
        print("\nWeights and Biases:")
        print(f"  W1 (Input to Hidden):\n{self.W1}")
        print(f"  b1 (Hidden biases): {self.b1}")
        print(f"  W2 (Hidden to Output):\n{self.W2}")
        print(f"  b2 (Output bias): {self.b2}")
        
        # Calculate and display total parameters
        total_params = self.W1.size + self.b1.size + self.W2.size + self.b2.size
        print(f"\nTotal Parameters: {total_params}")
        print(f"  - W1: {self.W1.size} (2×2)")
        print(f"  - b1: {self.b1.size} (1×2)")
        print(f"  - W2: {self.W2.size} (2×1)")
        print(f"  - b2: {self.b2.size} (1×1)")


# Training and testing function
def main():
    """Main function to run the XOR neural network"""
    print("XOR Neural Network with Backpropagation")
    print("=" * 60)
    
    # Create neural network
    nn = XORNeuralNetwork()
    
    # Display initial network information
    nn.display_network_info()
    
    # Train the network
    print("\n" + "=" * 50)
    print("Training Phase")
    print("=" * 50)
    losses = nn.train(epochs=10000, learning_rate=0.5, verbose=True)
    
    # Display trained network information
    nn.display_network_info()
    
    # Test the network
    accuracy, test_results = nn.test()
    
    # Demonstrate learning capability
    print("\n" + "=" * 50)
    print("Learning Demonstration")
    print("=" * 50)
    
    print("\nHow the network learns XOR:")
    print("1. Input (0,0): Network learns to output 0")
    print("2. Input (0,1): Network learns to output 1")
    print("3. Input (1,0): Network learns to output 1")
    print("4. Input (1,1): Network learns to output 0")
    
    print("\nKey Points:")
    print("- XOR is not linearly separable (cannot be solved by single-layer perceptron)")
    print("- Hidden layer creates non-linear transformation of inputs")
    print("- With 2 hidden neurons, network can learn XOR function")
    print(f"- Final accuracy: {accuracy:.2f}%")
    
    # Interactive prediction
    print("\n" + "=" * 50)
    print("Interactive Prediction")
    print("=" * 50)
    print("Enter two binary values (0 or 1) for XOR prediction")
    print("Type 'exit' to quit")
    
    while True:
        try:
            user_input = input("\nEnter values (e.g., '0 1'): ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting program.")
                break
            
            # Parse input
            values = user_input.split()
            if len(values) != 2:
                print("Please enter exactly 2 values!")
                continue
            
            # Convert to integers
            x1, x2 = int(values[0]), int(values[1])
            
            # Validate binary input
            if x1 not in [0, 1] or x2 not in [0, 1]:
                print("Please enter binary values (0 or 1)!")
                continue
            
            # Make prediction
            binary_pred, prob = nn.predict(np.array([x1, x2]))
            print(f"XOR({x1}, {x2}) = {binary_pred} (Confidence: {prob:.4f})")
            
            # Explain the result
            expected_xor = x1 ^ x2  # Python XOR operator
            if binary_pred == expected_xor:
                print(f"✓ Correct! Expected: {expected_xor}")
            else:
                print(f"✗ Incorrect! Expected: {expected_xor}")
                
        except ValueError:
            print("Invalid input! Please enter integers.")
        except KeyboardInterrupt:
            print("\nProgram interrupted.")
            break
    
    return nn, losses, accuracy


if __name__ == "__main__":
    # Run the neural network
    trained_nn, loss_history, final_accuracy = main()
    
    # Additional analysis (optional)
    print("\n" + "=" * 50)
    print("Training Analysis")
    print("=" * 50)
    print(f"Initial loss: {loss_history[0]:.6f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Loss reduction: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.2f}%")
    
    # Check if network learned XOR perfectly
    if final_accuracy == 100:
        print("\n✅ SUCCESS: Network learned XOR function perfectly!")
    else:
        print("\n⚠️ Network did not achieve perfect accuracy.")
        print("Try increasing epochs or adjusting learning rate.")
