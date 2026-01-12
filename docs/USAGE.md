# Usage Guide

## Basic Usage

### Using the HeimdallDetector Class

```python
from heimdall import HeimdallDetector

# Initialize detector
detector = HeimdallDetector(model_name="mistralai/Mistral-7B-v0.3")

# Analyze text
result = detector.analyze("The capital of France is Paris.")
print(f"R-Score: {result['r_score']:.4f}")
print(f"Classification: {result['classification']}")
```

### Command Line

```bash
# Run optimized test (GPU required)
python tests/test_optimized.py

# Run basic test (CPU compatible)
python tests/test_gpt2.py

# Run HaluEval benchmark
python tests/benchmark_halueval.py
```

## Configuration Options

### Model Selection

```python
# GPT-2 (CPU friendly)
detector = HeimdallDetector(model_name="gpt2")

# Mistral 7B (recommended)
detector = HeimdallDetector(model_name="mistralai/Mistral-7B-v0.3")

# Llama 3.1 (requires access)
detector = HeimdallDetector(model_name="meta-llama/Meta-Llama-3.1-8B")
```

### Quantization

```python
# Enable 4-bit quantization (reduces VRAM usage)
detector = HeimdallDetector(
    model_name="mistralai/Mistral-7B-v0.3",
    use_quantization=True
)
```

### Target Layer

```python
# Specify target layer for analysis
detector = HeimdallDetector(
    model_name="mistralai/Mistral-7B-v0.3",
    target_layer=24  # Optimal for 7B models
)
```

## Analysis Modes

### Single-Head Analysis (Fast)

```python
result = detector.analyze(text, multihead=False)
```

- ~2x faster
- Good for real-time applications

### Multi-Head Analysis (Accurate)

```python
result = detector.analyze(text, multihead=True)
```

- +15% precision
- Detects dominant vortex
- Best for research

## Output Format

```python
result = {
    'r_score': 0.324,           # Coherence ratio
    'classification': 'factual', # or 'hallucinated'
    'confidence': 0.87,          # Confidence score
    'head_scores': {...},        # Per-head R-scores
    'layer': 24                  # Analyzed layer
}
```

## Batch Processing

```python
texts = [
    "The Earth orbits the Sun.",
    "The Moon is made of cheese.",
]

results = detector.analyze_batch(texts)
for text, result in zip(texts, results):
    print(f"{text[:50]}... -> {result['classification']}")
```

## Integration Example

```python
from heimdall import HeimdallDetector

class SafeGenerator:
    def __init__(self):
        self.detector = HeimdallDetector()

    def generate_safe(self, prompt):
        response = self.llm.generate(prompt)
        result = self.detector.analyze(response)

        if result['classification'] == 'hallucinated':
            return self.regenerate_with_rag(prompt)
        return response
```
