#!/usr/bin/env python3
"""
JAX JIT Neural Network with Interactive Training Data Input
============================================================

Enhanced version with optional interactive mode for custom training data.
"""

import time
import json
from typing import Tuple, Dict, Any, List, Optional
import jax
import jax.numpy as jnp
from jax import grad, jit, make_jaxpr
from jax import random
import numpy as np
import argparse

print(f"JAX version: {jax.__version__}")
print()

# ============================================================================
# Data Input Utilities
# ============================================================================

def parse_numbers_input(prompt: str, expected_count: Optional[int] = None) -> List[float]:
    """
    Parse comma-separated numbers from user input.
    
    Args:
        prompt: Prompt to display to user
        expected_count: If set, validate exact number of values
    
    Returns:
        List of floats
    """
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                return []
            
            # Parse comma-separated values
            values = [float(x.strip()) for x in user_input.split(',')]
            
            if expected_count and len(values) != expected_count:
                print(f"  Error: Expected {expected_count} values, got {len(values)}")
                print(f"  Please enter exactly {expected_count} comma-separated numbers")
                continue
            
            return values
        except ValueError as e:
            print(f"  Error: Invalid input. Please enter numbers separated by commas.")
            print(f"  Example: 1.5, 2.3, -0.7, 4.2")
        except KeyboardInterrupt:
            print("\n\nInput cancelled.")
            return []

def interactive_data_input() -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray], int, int]:
    """
    Interactive mode: Allow user to input custom training data.
    
    Returns:
        X: Input features array or None if cancelled
        y: Target values array or None if cancelled  
        input_dim: Number of input features
        n_classes: Number of output classes
    """
    print("\n" + "="*70)
    print("INTERACTIVE TRAINING DATA INPUT")
    print("="*70)
    print()
    print("You can input your own training data for the neural network.")
    print("The network will learn to map your inputs to your desired outputs.")
    print()
    
    try:
        # Get dimensions
        print("Step 1: Configure data dimensions")
        print("-" * 70)
        
        while True:
            try:
                input_dim = int(input("Number of input features (e.g., 3): ").strip())
                if input_dim < 1 or input_dim > 100:
                    print("  Please enter a value between 1 and 100")
                    continue
                break
            except ValueError:
                print("  Please enter a valid integer")
        
        while True:
            try:
                n_classes = int(input("Number of output classes (e.g., 2): ").strip())
                if n_classes < 2 or n_classes > 10:
                    print("  Please enter a value between 2 and 10")
                    continue
                break
            except ValueError:
                print("  Please enter a valid integer")
        
        print()
        print(f"Configuration: {input_dim} inputs → {n_classes} outputs")
        print()
        
        # Get training samples
        print("Step 2: Enter training samples")
        print("-" * 70)
        print("Enter each training sample as comma-separated numbers.")
        print("Format: input1, input2, ..., inputN, output_class")
        print(f"Example with {input_dim} inputs and {n_classes} classes:")
        example_inputs = ", ".join([f"{i+1}.0" for i in range(input_dim)])
        print(f"  {example_inputs}, 0  (for class 0)")
        print(f"  {example_inputs}, 1  (for class 1)")
        print()
        print("Enter blank line when finished.")
        print()
        
        samples = []
        sample_num = 1
        
        while True:
            prompt = f"Sample {sample_num} ({input_dim + 1} values): "
            values = parse_numbers_input(prompt)
            
            if not values:
                if sample_num == 1:
                    print("  No samples entered. Using default dummy data.")
                    return None, None, input_dim, n_classes
                break
            
            if len(values) != input_dim + 1:
                print(f"  Error: Expected {input_dim + 1} values "
                      f"({input_dim} inputs + 1 output), got {len(values)}")
                continue
            
            # Validate output class
            output_class = int(values[-1])
            if output_class < 0 or output_class >= n_classes:
                print(f"  Error: Output class must be between 0 and {n_classes-1}")
                continue
            
            samples.append(values)
            sample_num += 1
            
            if sample_num > 1000:
                print("  Maximum 1000 samples reached.")
                break
        
        if not samples:
            return None, None, input_dim, n_classes
        
        # Convert to arrays
        samples = np.array(samples)
        X = jnp.array(samples[:, :input_dim], dtype=jnp.float32)
        y_indices = samples[:, -1].astype(int)
        y = jnp.eye(n_classes)[y_indices]
        
        print()
        print(f"✓ Loaded {len(samples)} training samples")
        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {y.shape}")
        print()
        
        # Show sample of data
        print("First 3 samples:")
        for i in range(min(3, len(samples))):
            inputs = ", ".join([f"{x:.2f}" for x in X[i]])
            output = int(jnp.argmax(y[i]))
            print(f"  [{inputs}] → class {output}")
        
        if len(samples) > 3:
            print(f"  ... and {len(samples) - 3} more")
        
        return X, y, input_dim, n_classes
        
    except KeyboardInterrupt:
        print("\n\nInteractive input cancelled. Using default data.")
        return None, None, 20, 3
    except Exception as e:
        print(f"\nError during input: {e}")
        print("Using default data instead.")
        return None, None, 20, 3

def test_with_custom_input(params: list, input_dim: int, n_classes: int):
    """
    Allow user to test the trained model with custom inputs.
    """
    print("\n" + "="*70)
    print("TEST THE TRAINED MODEL")
    print("="*70)
    print()
    print("Enter test inputs to see what the trained model predicts.")
    print("Enter blank line to skip testing.")
    print()
    
    try:
        while True:
            values = parse_numbers_input(
                f"Test input ({input_dim} values, comma-separated): ",
                expected_count=input_dim
            )
            
            if not values:
                break
            
            # Make prediction
            x = jnp.array([values], dtype=jnp.float32)
            logits = forward_pass(params, x)
            probs = jax.nn.softmax(logits[0])
            predicted_class = int(jnp.argmax(probs))
            confidence = float(probs[predicted_class])
            
            print()
            print(f"  Input:    [{', '.join(f'{v:.2f}' for v in values)}]")
            print(f"  Output logits: [{', '.join(f'{v:.3f}' for v in logits[0])}]")
            print(f"  Probabilities: [{', '.join(f'{p:.1%}' for p in probs)}]")
            print(f"  → Predicted class: {predicted_class} (confidence: {confidence:.1%})")
            print()
            
            # Show all class probabilities
            print("  Class probabilities:")
            for i, prob in enumerate(probs):
                bar = "█" * int(prob * 30)
                print(f"    Class {i}: {bar} {prob:.1%}")
            print()
    
    except KeyboardInterrupt:
        print("\n\nTesting cancelled.")
    except Exception as e:
        print(f"\nError during testing: {e}")

# ============================================================================
# Neural Network (from previous version)
# ============================================================================

def init_mlp_params(layer_sizes: list, key: Any) -> list:
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        scale = jnp.sqrt(2.0 / in_size)
        w_key, b_key = random.split(keys[i])
        weights = random.normal(w_key, (in_size, out_size)) * scale
        biases = jnp.zeros(out_size)
        params.append((weights, biases))
    return params

def relu(x):
    return jnp.maximum(0, x)

def forward_pass(params: list, x: jnp.ndarray) -> jnp.ndarray:
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = relu(x)
    w_out, b_out = params[-1]
    return jnp.dot(x, w_out) + b_out

def loss_fn(params: list, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    predictions = forward_pass(params, x)
    return jnp.mean((predictions - y) ** 2)

forward_pass_jit = jit(forward_pass)
loss_fn_jit = jit(loss_fn)

@jit
def train_step(params: list, x: jnp.ndarray, y: jnp.ndarray, learning_rate: float):
    grads = grad(loss_fn)(params, x, y)
    updated_params = [
        (w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]
    loss = loss_fn(params, x, y)
    return updated_params, loss

def generate_dummy_data(n_samples: int = 1000, input_dim: int = 10, 
                       n_classes: int = 3, key: Any = None):
    if key is None:
        key = random.PRNGKey(0)
    key1, key2 = random.split(key)
    X = random.normal(key1, (n_samples, input_dim))
    y_indices = jnp.argmax(X[:, :n_classes], axis=1)
    y = jnp.eye(n_classes)[y_indices]
    return X, y

# ============================================================================
# Main with Interactive Mode
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='JAX Neural Network with optional interactive training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with default dummy data (original behavior)
  python jax_interactive.py
  
  # Interactive mode - input your own training data
  python jax_interactive.py --interactive
  
  # Interactive mode with custom architecture
  python jax_interactive.py -i --hidden 64 32 --epochs 100
        '''
    )
    
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode: input custom training data')
    parser.add_argument('--hidden', nargs='+', type=int, default=[64, 32],
                       help='Hidden layer sizes (default: 64 32)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("JAX NEURAL NETWORK - INTERACTIVE TRAINING")
    print("=" * 70)
    print()
    
    # Get training data
    if args.interactive:
        X_train, y_train, input_dim, n_classes = interactive_data_input()
        
        if X_train is None:
            print("Using default dummy data...")
            key = random.PRNGKey(42)
            X_train, y_train = generate_dummy_data(
                n_samples=1500, 
                input_dim=20, 
                n_classes=3, 
                key=key
            )
            input_dim, n_classes = 20, 3
    else:
        # Default behavior - use dummy data
        print("Using default dummy dataset...")
        print("(Use --interactive flag to input your own data)")
        print()
        key = random.PRNGKey(42)
        X_train, y_train = generate_dummy_data(
            n_samples=1500,
            input_dim=20,
            n_classes=3,
            key=key
        )
        input_dim, n_classes = 20, 3
    
    # Generate test data
    key = random.PRNGKey(123)
    X_test, y_test = generate_dummy_data(
        n_samples=500,
        input_dim=input_dim,
        n_classes=n_classes,
        key=key
    )
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    print(f"Input dimension: {input_dim}")
    print(f"Output classes: {n_classes}")
    print()
    
    # Initialize model
    layer_sizes = [input_dim] + args.hidden + [n_classes]
    key = random.PRNGKey(0)
    params = init_mlp_params(layer_sizes, key)
    
    total_params = sum(w.size + b.size for w, b in params)
    print(f"Model architecture: {' → '.join(map(str, layer_sizes))}")
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Train
    print("Training...")
    print("-" * 70)
    
    n_train = X_train.shape[0]
    n_batches = max(1, n_train // args.batch_size)
    
    for epoch in range(args.epochs):
        # Shuffle
        perm = np.random.permutation(n_train)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        epoch_loss = 0.0
        for i in range(n_batches):
            start = i * args.batch_size
            end = min(start + args.batch_size, n_train)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            params, batch_loss = train_step(params, X_batch, y_batch, args.lr)
            epoch_loss += batch_loss
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            avg_loss = epoch_loss / n_batches
            test_loss = loss_fn_jit(params, X_test, y_test)
            print(f"Epoch {epoch:3d}: train_loss={avg_loss:.4f}, test_loss={test_loss:.4f}")
    
    print("-" * 70)
    print("✓ Training complete!")
    print()
    
    # Final evaluation
    final_loss = loss_fn_jit(params, X_test, y_test)
    print(f"Final test loss: {final_loss:.4f}")
    print()
    
    # Interactive testing
    if args.interactive:
        test_with_custom_input(params, input_dim, n_classes)
    else:
        print("Tip: Run with --interactive to test with your own inputs!")
        print()
    
    print("=" * 70)
    print("Done! The model has learned to map your inputs to outputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
