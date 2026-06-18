# JAX JIT Neural Network Example

This repository contains a demonstration of JAX's JIT compilation capabilities, inspired by the Hacker News discussion about tracing compilers and JAX.

## Hacker News Discussion Context

**Original Discussion:** https://news.ycombinator.com/item?id=48583606

Key insights from the discussion:
- **Trace compilation** is generally considered a "dead end" for general-purpose dynamic languages
- **JAX succeeds** where others fail because numerical computing has stable, predictable control flow
- Unlike TraceMonkey or LuaJIT which struggled with branchy code, JAX works well because:
  - ML workloads have regular, predictable patterns
  - Control flow is stable (enables GPU parallelization)
  - The domain is narrow and well-behaved
- JAX uses **tracing + XLA compilation** to achieve high performance
- Combines **automatic differentiation** with **JIT compilation**

## What This Example Demonstrates

The `jax_jit_example.py` file shows:

1. **JIT-compiled forward pass** - Using `@jit` decorator
2. **Automatic differentiation** - Using `grad()` for backward pass
3. **Compiled training step** - Entire train step is JIT-compiled
4. **Performance comparison** - Python vs JIT-compiled code
5. **Simple MLP** - 2-layer neural network with ReLU activations

## Key JAX Concepts Demonstrated

### 1. Just-In-Time Compilation
```python
@jit
def forward_pass(params, x):
    # This function is traced on first call
    # and compiled to optimized code
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = relu(x)
    return x
```

### 2. Automatic Differentiation
```python
# Compute gradients automatically
grads = grad(loss_fn)(params, x, y)

# JAX traces the computation and generates
# efficient backward pass code
```

### 3. Whole-Function Optimization
```python
@jit
def train_step(params, x, y, learning_rate):
    grads = grad(loss_fn)(params, x, y)  # Backward pass
    # Parameter updates...
    return updated_params, loss  # Everything compiled together!
```

## Why JAX's Tracing Works

From the HN discussion:

> "ML compilers in particular go beyond even the level of stability you would expect from numerical programs. Due to how the SIMT model works, the hardware heavily punishes unstable branches... Consequently, the whole stack is built around the expectation of stable control flow"

**JAX succeeds because:**
- ✅ Predictable control flow (no dynamic Python features in jitted code)
- ✅ Static shapes (arrays have fixed dimensions)
- ✅ Pure functions (no side effects)
- ✅ Numerical operations (matrix math, not string processing)

**General-purpose tracing JITs struggle because:**
- ❌ Arbitrary Python dynamism
- ❌ Unpredictable control flow
- ❌ Dynamic types and shapes
- ❌ Side effects everywhere

## Running the Example

### Installation
```bash
pip install jax jaxlib
```

### For GPU support:
```bash
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Run the demo:
```bash
python jax_jit_example.py
```

### Expected Output:
```
JAX version: 0.4.x
JAX devices: [CpuDevice(id=0)]
JAX backend: cpu

============================================================
JAX JIT NEURAL NETWORK DEMO
============================================================

Training for 50 epochs...
Epoch   0 | Train Loss: 0.4521 | Test Loss: 0.4234 | Test Acc: 0.6520
Epoch  10 | Train Loss: 0.2134 | Test Loss: 0.2241 | Test Acc: 0.8340
...
Epoch  49 | Train Loss: 0.0892 | Test Loss: 0.1023 | Test Acc: 0.9260

PERFORMANCE BENCHMARK:
  Python:     2.345s
  JIT:        0.123s
  Speedup:    19.1x
```

## Performance Characteristics

**First call:** Slow (includes compilation time)
- JAX traces the Python function
- Builds computational graph
- Compiles with XLA to native code

**Subsequent calls:** Very fast (uses cached compilation)
- Reuses compiled code
- No Python overhead
- Optimized for specific shapes/dtypes

**When recompilation happens:**
- Different input shapes
- Different dtypes
- Different static arguments

## Files in This Repository

- `jax_jit_example.py` - Main example with MLP training
- `README.md` - This file

## Further Reading

- **JAX Documentation:** https://docs.jax.dev/
- **JAX JIT Guide:** https://docs.jax.dev/en/latest/jit.html
- **Original HN Discussion:** https://news.ycombinator.com/item?id=48583606
- **XLA Documentation:** https://www.tensorflow.org/xla

## Key Takeaways from HN Discussion

1. **Tracing is not dead** - It works great in narrow domains like numerical computing
2. **JAX proves** that with the right constraints, tracing JITs can be very successful
3. **Domain matters** - Stable control flow is essential for tracing to work well
4. **ML workloads are ideal** - Predictable, numerical, and benefit greatly from compilation
5. **Trade-offs exist** - JAX gives up Python dynamism for performance

## License

MIT License - Feel free to use this example for learning JAX!
