#!/usr/bin/env python3
"""
JAX JIT Neural Network Example
================================

This example demonstrates JAX's JIT compilation capabilities for a simple
neural network, inspired by the Hacker News discussion about tracing compilers
and JAX (https://news.ycombinator.com/item?id=48583606).

Key Concepts from HN Discussion:
- JAX is a tracing JIT compiler that works well for numerical computing
- Unlike general-purpose tracing JITs (TraceMonkey, LuaJIT), JAX succeeds because
  numerical programs have stable, predictable control flow
- JAX uses XLA (Accelerated Linear Algebra) to compile to CPU/GPU/TPU
- Tracing works well when control flow is stable - perfect for ML workloads

This example shows:
1. JIT-compiled forward pass
2. Automatic differentiation for backward pass
3. Performance comparison: Python vs JIT
4. Simple 2-layer MLP for demonstration
"""

import time
from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")
print()

# ============================================================================
# Neural Network Components
# ============================================================================

def init_mlp_params(layer_sizes: list, key: Any) -> list:
    """
    Initialize MLP parameters with He initialization.
    
    Args:
        layer_sizes: List of layer dimensions [input, hidden1, ..., output]
        key: JAX random key
    
    Returns:
        List of (weights, biases) tuples for each layer
    """
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    
    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        # He initialization for ReLU networks
        scale = jnp.sqrt(2.0 / in_size)
        w_key, b_key = random.split(keys[i])
        
        weights = random.normal(w_key, (in_size, out_size)) * scale
        biases = jnp.zeros(out_size)
        
        params.append((weights, biases))
    
    return params

def relu(x):
    """ReLU activation function"""
    return jnp.maximum(0, x)

def forward_pass(params: list, x: jnp.ndarray) -> jnp.ndarray:
    """
    Forward pass through MLP.
    
    Args:
        params: List of (weights, biases) tuples
        x: Input array [batch_size, input_dim]
    
    Returns:
        Output predictions
    """
    # Hidden layers with ReLU
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = relu(x)
    
    # Output layer (no activation)
    w_out, b_out = params[-1]
    logits = jnp.dot(x, w_out) + b_out
    
    return logits

def loss_fn(params: list, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute mean squared error loss.
    
    Args:
        params: Network parameters
        x: Input data
        y: Target values
    
    Returns:
        Scalar loss value
    """
    predictions = forward_pass(params, x)
    return jnp.mean((predictions - y) ** 2)

def accuracy(params: list, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute accuracy for classification"""
    predictions = forward_pass(params, x)
    predicted_classes = jnp.argmax(predictions, axis=1)
    target_classes = jnp.argmax(y, axis=1)
    return jnp.mean(predicted_classes == target_classes)

# ============================================================================
# JIT-Compiled Versions
# ============================================================================

# JIT-compile the forward pass
# The first call traces the function and compiles it
# Subsequent calls use the compiled version (much faster!)
forward_pass_jit = jit(forward_pass)

# JIT-compile the loss function
loss_fn_jit = jit(loss_fn)

# Create a JIT-compiled gradient function
# This is where JAX shines - automatic differentiation + JIT compilation
grad_fn = jit(grad(loss_fn))

# JIT-compiled training step
@jit
def train_step(params: list, x: jnp.ndarray, y: jnp.ndarray, learning_rate: float):
    """
    Single training step: compute gradients and update parameters.
    
    This entire function is JIT-compiled, including:
    - Forward pass
    - Loss computation
    - Backward pass (via autodiff)
    - Parameter updates
    
    Args:
        params: Current parameters
        x: Input batch
        y: Target batch
        learning_rate: Learning rate for SGD
    
    Returns:
        Updated parameters and loss value
    """
    # Compute gradients (automatic differentiation)
    grads = grad(loss_fn)(params, x, y)
    
    # Update parameters (SGD)
    # grads is a list of (dW, db) tuples matching params structure
    updated_params = [
        (w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]
    
    # Compute loss for monitoring
    loss = loss_fn(params, x, y)
    
    return updated_params, loss

# ============================================================================
# Training Utilities
# ============================================================================

def generate_dummy_data(n_samples: int = 1000, 
                       input_dim: int = 10, 
                       n_classes: int = 3,
                       key: Any = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate dummy classification dataset"""
    if key is None:
        key = random.PRNGKey(0)
    
    key1, key2 = random.split(key)
    
    # Generate random features
    X = random.normal(key1, (n_samples, input_dim))
    
    # Generate labels based on a simple rule (for demo purposes)
    # Class is determined by which quadrant the point falls into
    y_indices = jnp.argmax(X[:, :n_classes], axis=1)
    y = jnp.eye(n_classes)[y_indices]
    
    return X, y

def train_model(params: list, 
                X_train: jnp.ndarray, 
                y_train: jnp.ndarray,
                X_test: jnp.ndarray,
                y_test: jnp.ndarray,
                num_epochs: int = 100,
                batch_size: int = 32,
                learning_rate: float = 0.01) -> Tuple[list, Dict[str, list]]:
    """
    Train the model with mini-batch SGD.
    
    Args:
        params: Initial parameters
        X_train, y_train: Training data
        X_test, y_test: Test data
        num_epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Learning rate
    
    Returns:
        Trained parameters and training history
    """
    n_train = X_train.shape[0]
    n_batches = n_train // batch_size
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    print(f"Training for {num_epochs} epochs ({n_batches} batches/epoch)...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Shuffle training data
        perm = np.random.permutation(n_train)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        epoch_loss = 0.0
        
        # Mini-batch training
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # JIT-compiled training step
            params, batch_loss = train_step(params, X_batch, y_batch, learning_rate)
            epoch_loss += batch_loss
        
        # Evaluate
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            avg_loss = epoch_loss / n_batches
            test_loss = loss_fn_jit(params, X_test, y_test)
            test_acc = accuracy(params, X_test, y_test)
            
            history['train_loss'].append(float(avg_loss))
            history['test_loss'].append(float(test_loss))
            history['test_accuracy'].append(float(test_acc))
            
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.4f}")
    
    print("-" * 60)
    return params, history

# ============================================================================
# Performance Comparison
# ============================================================================

def benchmark_forward_pass():
    """Compare Python vs JIT-compiled forward pass performance"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK: Python vs JIT")
    print("=" * 60)
    
    # Initialize small network
    key = random.PRNGKey(42)
    params = init_mlp_params([100, 64, 32, 10], key)
    
    # Create test input
    x = random.normal(random.PRNGKey(1), (1000, 100))
    
    # Warm up JIT (first call includes compilation time)
    print("\nWarming up JIT compiler...")
    _ = forward_pass_jit(params, x[:1])
    _ = forward_pass_jit(params, x[:1])  # Second call uses cached compilation
    
    # Benchmark Python version
    print("\nBenchmarking Python version...")
    start = time.time()
    for _ in range(100):
        _ = forward_pass(params, x)
    python_time = time.time() - start
    
    # Benchmark JIT version
    print("Benchmarking JIT-compiled version...")
    start = time.time()
    for _ in range(100):
        _ = forward_pass_jit(params, x)
    jit_time = time.time() - start
    
    print(f"\nResults (100 iterations, batch size 1000):")
    print(f"  Python:     {python_time:.3f}s")
    print(f"  JIT:        {jit_time:.3f}s")
    print(f"  Speedup:    {python_time/jit_time:.1f}x")
    print(f"  Time saved: {python_time - jit_time:.3f}s")
    
    return python_time, jit_time

# ============================================================================
# Main Demo
# ============================================================================

def main():
    print("=" * 60)
    print("JAX JIT NEURAL NETWORK DEMO")
    print("=" * 60)
    print()
    print("This demo shows JAX's tracing JIT compiler in action.")
    print("Unlike general-purpose tracing JITs that struggle with")
    print("branchy code, JAX excels at numerical computing where")
    print("control flow is stable and predictable.")
    print()
    
    # Generate dummy dataset
    print("Generating dummy dataset...")
    key = random.PRNGKey(0)
    X, y = generate_dummy_data(n_samples=2000, input_dim=20, n_classes=3, key=key)
    
    # Train/test split
    split_idx = 1500
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Input dim: {X_train.shape[1]}, Classes: {y_train.shape[1]}")
    print()
    
    # Initialize model
    print("Initializing 2-layer MLP...")
    key, subkey = random.split(key)
    layer_sizes = [20, 64, 32, 3]  # input -> hidden1 -> hidden2 -> output
    params = init_mlp_params(layer_sizes, subkey)
    
    total_params = sum(w.size + b.size for w, b in params)
    print(f"Architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Train model
    trained_params, history = train_model(
        params,
        X_train, y_train,
        X_test, y_test,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.01
    )
    
    # Final evaluation
    final_train_loss = loss_fn_jit(trained_params, X_train, y_train)
    final_test_loss = loss_fn_jit(trained_params, X_test, y_test)
    final_test_acc = accuracy(trained_params, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Final Train Loss: {final_train_loss:.4f}")
    print(f"Final Test Loss:  {final_test_loss:.4f}")
    print(f"Final Test Acc:   {final_test_acc:.4f}")
    print()
    
    # Performance benchmark
    benchmark_forward_pass()
    
    print()
    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. JAX traces Python functions and compiles them with XLA")
    print("2. @jit decorator enables whole-function optimization")
    print("3. grad() provides automatic differentiation")
    print("4. Combining jit + grad gives compiled backward passes")
    print("5. Works best when control flow is stable (like this MLP)")
    print("6. First call is slow (compilation), subsequent calls are fast")
    print()
    print("This is why JAX succeeds where general tracing JITs struggle:")
    print("- Numerical code has predictable control flow")
    print("- No need to handle arbitrary Python dynamism")
    print("- Can make strong assumptions for optimization")
    print("=" * 60)


if __name__ == "__main__":
    main()
