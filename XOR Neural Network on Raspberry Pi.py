"""
XOR Neural Network on Raspberry Pi
Correct implementation with proper weights and English code
"""

import RPi.GPIO as GPIO
import numpy as np
import time

# ============================================
# 1. CORRECT XOR NEURAL NETWORK WEIGHTS
# ============================================

# Input to hidden layer weights (2×2)
weights_input_hidden = np.array([
    [ 5.0, -5.0],   # To hidden neuron 1
    [-5.0,  5.0]    # To hidden neuron 2
])

# Hidden to output layer weights (2×1)
weights_hidden_output = np.array([
    [ 5.0],         # From hidden neuron 1
    [ 5.0]          # From hidden neuron 2
])

# Biases
bias_hidden = np.array([-2.0, -2.0])   # Hidden layer bias
bias_output = np.array([-2.5])         # Output layer bias

# ============================================
# 2. GPIO PIN CONFIGURATION
# ============================================

# Define GPIO pins
BUTTON_A = 17     # Input button A
BUTTON_B = 27     # Input button B
LED_OUTPUT = 18   # Output LED (shows XOR result)
LED_STATUS = 22   # Status LED (always ON when running)

# ============================================
# 3. NEURAL NETWORK FUNCTIONS
# ============================================

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def predict_xor(input_a, input_b):
    """
    Predict XOR output using neural network
    
    Args:
        input_a: 0 or 1
        input_b: 0 or 1
    
    Returns:
        0 or 1
    """
    # Convert inputs to 2D array
    inputs = np.array([[input_a, input_b]])
    
    # Forward pass to hidden layer
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    
    # Forward pass to output layer
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)
    
    # Return binary output (threshold at 0.5)
    return 1 if final_output[0][0] > 0.5 else 0

# ============================================
# 4. TEST FUNCTIONS
# ============================================

def test_neural_network():
    """Test XOR neural network with all possible inputs"""
    print("\n🧪 Testing XOR Neural Network:")
    print("=" * 50)
    print("Input A | Input B | Expected | Predicted | Status")
    print("-" * 50)
    
    test_cases = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0)
    ]
    
    all_correct = True
    for a, b, expected in test_cases:
        predicted = predict_xor(a, b)
        status = "✓ PASS" if predicted == expected else "✗ FAIL"
        if predicted != expected:
            all_correct = False
        
        print(f"   {a}    |    {b}    |    {expected}    |    {predicted}    | {status}")
    
    print("=" * 50)
    return all_correct

def show_manual_calculation():
    """Show manual calculation for verification"""
    print("\n📊 Manual Calculation for A=0, B=1:")
    print("-" * 40)
    
    a, b = 0, 1
    print(f"Case: A={a}, B={b}")
    
    # Calculate hidden layer
    h1_input = a*5.0 + b*(-5.0) - 2.0
    h2_input = a*(-5.0) + b*5.0 - 2.0
    
    h1_output = 1/(1+np.exp(-h1_input))
    h2_output = 1/(1+np.exp(-h2_input))
    
    print(f"  h1_input = {a}×5.0 + {b}×(-5.0) - 2.0 = {h1_input:.2f}")
    print(f"  h2_input = {a}×(-5.0) + {b}×5.0 - 2.0 = {h2_input:.2f}")
    print(f"  h1_output = sigmoid({h1_input:.2f}) = {h1_output:.4f}")
    print(f"  h2_output = sigmoid({h2_input:.2f}) = {h2_output:.4f}")
    
    # Calculate output
    output_input = h1_output*5.0 + h2_output*5.0 - 2.5
    output = 1/(1+np.exp(-output_input))
    
    print(f"  output_input = {h1_output:.4f}×5.0 + {h2_output:.4f}×5.0 - 2.5 = {output_input:.2f}")
    print(f"  output = sigmoid({output_input:.2f}) = {output:.4f}")
    print(f"  Result: {1 if output > 0.5 else 0}")
    print("-" * 40)

# ============================================
# 5. RASPBERRY PI HARDWARE FUNCTIONS
# ============================================

def setup_gpio():
    """Initialize GPIO pins"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Setup buttons with PULL_DOWN configuration
    GPIO.setup(BUTTON_A, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(BUTTON_B, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
    # Setup LEDs
    GPIO.setup(LED_OUTPUT, GPIO.OUT)
    GPIO.setup(LED_STATUS, GPIO.OUT)
    
    # Turn ON status LED
    GPIO.output(LED_STATUS, GPIO.HIGH)
    
    print("✅ GPIO initialized successfully")

def read_buttons():
    """Read current state of both buttons"""
    # With PULL_DOWN:
    # Button not pressed = GPIO.LOW = 0
    # Button pressed = GPIO.HIGH = 1
    button_a = 1 if GPIO.input(BUTTON_A) == GPIO.HIGH else 0
    button_b = 1 if GPIO.input(BUTTON_B) == GPIO.HIGH else 0
    
    return button_a, button_b

def cleanup_gpio():
    """Cleanup GPIO before exit"""
    GPIO.output(LED_OUTPUT, GPIO.LOW)
    GPIO.output(LED_STATUS, GPIO.LOW)
    GPIO.cleanup()
    print("✅ GPIO cleaned up")

# ============================================
# 6. MAIN PROGRAM
# ============================================

def main():
    print("=" * 60)
    print("XOR Neural Network - Raspberry Pi Implementation")
    print("=" * 60)
    
    # First, test the neural network logic
    if not test_neural_network():
        print("\n❌ Neural network test failed!")
        show_manual_calculation()
        return
    
    print("\n✅ Neural network test passed!")
    
    # Initialize hardware
    setup_gpio()
    
    # Display connection information
    print("\n" + "=" * 60)
    print("READY TO START - HARDWARE SETUP")
    try:
        while True:
            # Read button states
            button_a, button_b = read_buttons()
            
            # Get neural network prediction
            output = predict_xor(button_a, button_b)
            
            # Control output LED
            if output == 1:
                GPIO.output(LED_OUTPUT, GPIO.HIGH)
                led_status = "ON"
            else:
                GPIO.output(LED_OUTPUT, GPIO.LOW)
                led_status = "OFF"
            
            # Display current state
            print(f"\rInputs: A={button_a}, B={button_b} → XOR={output} (LED: {led_status})", end="")
            
            # Small delay to reduce CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Program stopped by user")
    
    finally:
        cleanup_gpio()

# ============================================
# 7. ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
