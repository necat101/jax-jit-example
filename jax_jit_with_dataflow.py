#!/usr/bin/env python3
"""
JAX JIT Neural Network with Data Flow Analysis
================================================

Enhanced version incorporating insights from HN discussion about
data flow analysis in compilers (https://news.ycombinator.com/item?id=48583606).

HN Discussion Insights:
- "Data flow" is a core compiler topic alongside dead code elimination,
  dominator analysis, and SSA form
- JAX's tracing compiler succeeds by analyzing data flow in numerical programs
- Data flow analysis tracks how values propagate through computations
- This is fundamental to optimization and is what enables JAX's performance

New Features in v2:
- Data flow tracking through network layers
- Activation statistics and distribution analysis
- Data dependency visualization
- Flow analysis showing how information propagates
- Connection to compiler data flow analysis concepts
"""

import time
from typing import Tuple, Dict, Any, List
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, make_jaxpr
from jax import random
import numpy as np

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print()

# ============================================================================
# Data Flow Analysis Utilities
# ============================================================================

class DataFlowTracker:
    """
    Tracks how data flows and transforms through the network.
    
    In compiler theory, data flow analysis determines:
    - Reaching definitions
    - Live variables
    - Available expressions
    - Very busy expressions
    
    Here we track:
    - Activation statistics at each layer
    - Data distribution changes
    - Information flow through the network
    - Gradient flow during backprop
    """
    
    def __init__(self):
        self.activations = {}
        self.statistics = {}
    
    def track_layer(self, name: str, data: jnp.ndarray):
        """Track activation statistics for a layer"""
        self.activations[name] = data
        self.statistics[name] = {
            'mean': float(jnp.mean(data)),
            'std': float(jnp.std(data)),
            'min': float(jnp.min(data)),
            'max': float(jnp.max(data)),
            'sparsity': float(jnp.mean(data == 0)),  # % of zeros (ReLU sparsity)
            'shape': data.shape,
        }
    
    def get_flow_report(self) -> str:
        """Generate data flow analysis report"""
        lines = ["\n" + "="*60]
        lines.append("DATA FLOW ANALYSIS")
        lines.append("="*60)
        lines.append("\nHow data transforms through the network:")
        lines.append("(Inspired by compiler data flow analysis)")
        lines.append("")
        
        for name, stats in self.statistics.items():
            lines.append(f"Layer: {name}")
            lines.append(f"  Shape: {stats['shape']}")
            lines.append(f"  Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f}")
            lines.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            lines.append(f"  Sparsity: {stats['sparsity']*100:.1f}% zeros")
            lines.append("")
        
        return "\n".join(lines)

def analyze_jaxpr(jaxpr):
    """
    Analyze JAX's internal representation (similar to compiler IR).
    
    JAX traces Python code into Jaxpr - a simple intermediate representation
    that makes data flow explicit. This is analogous to how compilers use
    SSA form to make data flow analysis tractable.
    """
    print("\n" + "="*60)
    print("JAXPR ANALYSIS (JAX's Intermediate Representation)")
    print("="*60)
    print("\nJAX traces your Python function and converts it to Jaxpr,")
    print("a simple functional IR where data flow is explicit.")
    print("\nThis is similar to how compilers use SSA (Static Single Assignment)")
    print("form to make data flow analysis easier.")
    print("\nJaxpr for forward_pass:")
    print("-" * 60)
    print(jaxpr)
    print("-" * 60)

# ============================================================================
# Enhanced Neural Network with Data Flow Tracking
# ============================================================================

def init_mlp_params(layer_sizes: list, key: Any) -> list:
    """Initialize MLP parameters with He initialization."""
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

def forward_pass_with_tracking(params: list, x: jnp.ndarray, 
                               tracker: DataFlowTracker = None) -> jnp.ndarray:
    """
    Forward pass with optional data flow tracking.
    
    This demonstrates data flow analysis - tracking how data transforms
    at each stage of computation, similar to what compilers do for optimization.
    """
    if tracker:
        tracker.track_layer("input", x)
    
    # Hidden layers
    for i, (w, b) in enumerate(params[:-1]):
        x = jnp.dot(x, w) + b
        if tracker:
            tracker.track_layer(f"linear_{i}", x)
        
        x = relu(x)
        if tracker:
            tracker.track_layer(f"relu_{i}", x)
    
    # Output layer
    w_out, b_out = params[-1]
    logits = jnp.dot(x, w_out) + b_out
    if tracker:
        tracker.track_layer("output", logits)
    
    return logits

def forward_pass(params: list, x: jnp.ndarray) -> jnp.ndarray:
    """Standard forward pass without tracking (for JIT compilation)"""
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = relu(x)
    
    w_out, b_out = params[-1]
    return jnp.dot(x, w_out) + b_out

def loss_fn(params: list, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    predictions = forward_pass(params, x)
    return jnp.mean((predictions - y) ** 2)

# JIT-compiled versions
forward_pass_jit = jit(forward_pass)
loss_fn_jit = jit(loss_fn)

@jit
def train_step(params: list, x: jnp.ndarray, y: jnp.ndarray, learning_rate: float):
    """JIT-compiled training step with gradient computation"""
    grads = grad(loss_fn)(params, x, y)
    updated_params = [
        (w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]
    loss = loss_fn(params, x, y)
    return updated_params, loss, grads

# ============================================================================
# Data Flow Visualization
# ============================================================================

def visualize_data_flow(params: list, x_sample: jnp.ndarray):
    """
    Visualize how data flows through the network.
    
    This is analogous to data flow analysis in compilers, where we track:
    - How values propagate
    - Where they're defined and used
    - How they transform through operations
    """
    print("\n" + "="*60)
    print("DATA FLOW VISUALIZATION")
    print("="*60)
    print("\nTracking a single input through the network...")
    print("(Similar to reaching definitions analysis in compilers)")
    print()
    
    tracker = DataFlowTracker()
    output = forward_pass_with_tracking(params, x_sample, tracker)
    
    print(tracker.get_flow_report())
    
    print("Data Flow Insights:")
    print("-" * 60)
    
    # Analyze how data transforms
    if 'input' in tracker.statistics and 'output' in tracker.statistics:
        input_stats = tracker.statistics['input']
        output_stats = tracker.statistics['output']
        
        print(f"• Input variance:  {input_stats['std']**2:.4f}")
        print(f"• Output variance: {output_stats['std']**2:.4f}")
        print(f"• Variance ratio:  {output_stats['std']**2 / (input_stats['std']**2 + 1e-8):.2f}x")
        print()
        print("• ReLU layers introduce sparsity (many zeros)")
        print("• Each layer transforms the data distribution")
        print("• Deep networks learn hierarchical representations")
        print()
        print("Compiler analogy:")
        print("- Each layer is like a basic block")
        print("- Activations are like SSA values")
        print("- Data flow analysis tracks how values propagate")
        print("- This enables optimizations like constant folding, CSE")
    
    return tracker

def analyze_gradient_flow(params: list, x: jnp.ndarray, y: jnp.ndarray):
    """
    Analyze how gradients flow backward through the network.
    
    This demonstrates backward data flow analysis - tracking how
    error signals propagate backwards, analogous to live variable
    analysis in compilers.
    """
    print("\n" + "="*60)
    print("GRADIENT FLOW ANALYSIS (Backward Data Flow)")
    print("="*60)
    print()
    print("Computing gradients to see how error flows backward...")
    print("(Similar to backward data flow analysis in compilers)")
    print()
    
    # Get gradients
    _, _, grads = train_step(params, x, y, 0.01)
    
    print("Gradient magnitudes by layer:")
    print("-" * 60)
    
    for i, (dw, db) in enumerate(grads):
        w_norm = jnp.linalg.norm(dw)
        b_norm = jnp.linalg.norm(db)
        
        layer_type = "Hidden" if i < len(grads) - 1 else "Output"
        print(f"Layer {i} ({layer_type}):")
        print(f"  Weight grad norm: {w_norm:.6f}")
        print(f"  Bias grad norm:   {b_norm:.6f}")
        
        # Check for vanishing/exploding gradients
        if w_norm < 1e-7:
            print(f"  ⚠️  Vanishing gradient detected!")
        elif w_norm > 10.0:
            print(f"  ⚠️  Exploding gradient detected!")
        print()
    
    print("This backward flow is:")
    print("• Computed via automatic differentiation")
    print("• JIT-compiled for efficiency")
    print("• Analogous to backward data flow analysis")
    print("• Tracks how errors propagate (like live variables)")

# ============================================================================
# Training Utilities
# ============================================================================

def generate_dummy_data(n_samples: int = 1000, input_dim: int = 10, 
                       n_classes: int = 3, key: Any = None):
    if key is None:
        key = random.PRNGKey(0)
    key1, key2 = random.split(key)
    X = random.normal(key1, (n_samples, input_dim))
    y_indices = jnp.argmax(X[:, :n_classes], axis=1)
    y = jnp.eye(n_classes)[y_indices]
    return X, y

def main():
    print("=" * 70)
    print("JAX JIT NEURAL NETWORK WITH DATA FLOW ANALYSIS")
    print("=" * 70)
    print()
    print("Enhanced version incorporating compiler data flow concepts")
    print("from HN discussion: https://news.ycombinator.com/item?id=48583606")
    print()
    print("Key Insight: JAX's tracing compiler works because it can")
    print("analyze data flow in numerical programs - similar to how")
    print("traditional compilers use data flow analysis for optimization.")
    print("=" * 70)
    print()
    
    # Generate data
    key = random.PRNGKey(0)
    X, y = generate_dummy_data(n_samples=2000, input_dim=20, n_classes=3, key=key)
    X_train, y_train = X[:1500], y[:1500]
    X_test, y_test = X[1500:], y[1500:]
    
    # Initialize model
    key, subkey = random.split(key)
    layer_sizes = [20, 64, 32, 3]
    params = init_mlp_params(layer_sizes, subkey)
    
    print(f"Model: {' -> '.join(map(str, layer_sizes))}")
    print(f"Parameters: {sum(w.size + b.size for w, b in params):,}")
    print()
    
    # Show JAXPR (JAX's IR)
    print("Analyzing JAX's intermediate representation...")
    sample_x = X_train[:2]
    jaxpr = make_jaxpr(lambda p, x: forward_pass(p, x))(params, sample_x)
    analyze_jaxpr(jaxpr)
    
    # Visualize data flow
    print("\nVisualizing data flow through untrained network...")
    visualize_data_flow(params, X_train[:5])
    
    # Quick training demo
    print("\n" + "="*70)
    print("TRAINING DEMO (5 epochs)")
    print("="*70)
    
    for epoch in range(5):
        params, loss, grads = train_step(params, X_train[:128], y_train[:128], 0.01)
        if epoch == 0 or epoch == 4:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    # Analyze gradient flow
    analyze_gradient_flow(params, X_test[:32], y_test[:32])
    
    # Final evaluation
    test_loss = loss_fn_jit(params, X_test, y_test)
    
    print("\n" + "="*70)
    print("SUMMARY: JAX, Data Flow, and Compilers")
    print("="*70)
    print()
    print("What we demonstrated:")
    print("1. ✅ JAX traces Python functions to Jaxpr (IR with explicit data flow)")
    print("2. ✅ Data flow analysis tracks how values transform through layers")
    print("3. ✅ Forward and backward data flow (activations and gradients)")
    print("4. ✅ JIT compilation optimizes based on data flow analysis")
    print()
    print("Connection to compiler theory (from HN discussion):")
    print("• Data flow analysis is fundamental to optimization")
    print("• JAX uses it to understand computation graphs")
    print("• Stable data flow enables effective tracing (unlike general Python)")
    print("• Similar to how compilers use SSA form and data flow analysis")
    print()
    print("Why this matters:")
    print("• Understanding data flow helps debug ML models")
    print("• Vanishing/exploding gradients show up in flow analysis")
    print("• Activation statistics reveal network behavior")
    print("• JAX's success proves tracing works in constrained domains")
    print()
    print("="*70)
    print(f"Final test loss: {test_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
