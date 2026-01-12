"""
HEIMDALL Detector - Core Module
Real-Time Hallucination Detection via Coherence Inversion
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gudhi
import numpy as np


class HeimdallDetector:
    """
    Topology-based Hallucination Detector

    Uses persistent homology to analyze attention patterns and detect
    hallucinations in Large Language Model outputs.

    The key discovery is "coherence inversion": hallucinated responses
    exhibit consistently higher topological coherence than factual ones.

    Usage:
        detector = HeimdallDetector(model_name="gpt2")
        result = detector.analyze("The Earth is flat.")
        if result['classification'] == 'hallucinated':
            print("Warning: Possible hallucination detected!")

    Attributes:
        model_name: HuggingFace model identifier
        use_quantization: Enable 4-bit quantization (requires CUDA)
        target_layer: Layer index for attention analysis (-1 = last)
    """

    def __init__(self, model_name="gpt2", use_quantization=False, target_layer=-1):
        """
        Initialize the detector

        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "mistralai/Mistral-7B-v0.3")
            use_quantization: Enable 4-bit quantization (requires CUDA)
            target_layer: Layer for attention analysis (-1 = last layer)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_layer = target_layer
        self.model_name = model_name
        self.use_quantization = use_quantization

        print(f"Initializing HEIMDALL...")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Quantization: {use_quantization}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with optional quantization
        if use_quantization and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                output_attentions=True,
                trust_remote_code=True,
                attn_implementation="eager"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
                trust_remote_code=True
            ).to(self.device)

        # Configure pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[OK] HEIMDALL ready!")

    def analyze(self, text, multihead=True, threshold=0.15):
        """
        Analyze text for hallucination indicators

        Args:
            text: Text to analyze
            multihead: Analyze all attention heads (recommended)
            threshold: R-score threshold for classification

        Returns:
            dict: Analysis results containing:
                - r_score: Coherence ratio (higher = more coherent)
                - classification: 'factual' or 'hallucinated'
                - confidence: Confidence score
                - layer: Analyzed layer index
                - head_scores: Per-head R-scores (if multihead=True)
        """
        if not text:
            return {
                'r_score': 1.0,
                'classification': 'factual',
                'confidence': 0.0,
                'layer': self.target_layer,
                'head_scores': {}
            }

        # Tokenization
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=False)

        # Extract attention from target layer
        attention_layer = outputs.attentions[self.target_layer][0].cpu().numpy()

        head_scores = {}

        if multihead:
            # Analyze all heads (dominant vortex detection)
            max_R = 0.0
            for head_idx in range(attention_layer.shape[0]):
                head_att = attention_layer[head_idx]
                R = self._compute_R_score(head_att)
                head_scores[f"head_{head_idx}"] = R
                max_R = max(max_R, R)
            r_score = max_R
        else:
            # Use head average
            mean_att = np.mean(attention_layer, axis=0)
            r_score = self._compute_R_score(mean_att)

        # Classification based on coherence inversion
        # Higher R-score indicates potential hallucination
        classification = 'hallucinated' if r_score > threshold else 'factual'

        # Confidence based on distance from threshold
        confidence = min(abs(r_score - threshold) / threshold, 1.0)

        return {
            'r_score': float(r_score),
            'classification': classification,
            'confidence': float(confidence),
            'layer': self.target_layer,
            'head_scores': head_scores
        }

    def analyze_batch(self, texts, multihead=True, threshold=0.15):
        """
        Analyze multiple texts in batch

        Args:
            texts: List of texts to analyze
            multihead: Analyze all attention heads
            threshold: R-score threshold for classification

        Returns:
            list: List of analysis results
        """
        return [self.analyze(text, multihead, threshold) for text in texts]

    def _compute_R_score(self, attention_matrix):
        """
        Compute R-Score via Persistent Homology

        The R-score measures topological coherence using H1 features
        (cycles/vortices) from cubical complex homology.

        Args:
            attention_matrix: 2D attention matrix

        Returns:
            float: R-score (max lifetime / feature count)
        """
        # Min-max normalization
        att_min = attention_matrix.min()
        att_max = attention_matrix.max()

        if att_max - att_min < 1e-10:
            return 0.0

        att_norm = (attention_matrix - att_min) / (att_max - att_min)

        try:
            # Cubical Complex for efficient computation
            cubical = gudhi.CubicalComplex(
                dimensions=attention_matrix.shape,
                top_dimensional_cells=att_norm.flatten()
            )
            cubical.compute_persistence()
            persistence = cubical.persistence_intervals_in_dimension(1)  # H1: vortices

            if len(persistence) > 0:
                lifetimes = persistence[:, 1] - persistence[:, 0]
                # Remove infinite lifetimes
                lifetimes = lifetimes[~np.isinf(lifetimes)]
                if len(lifetimes) > 0:
                    # R = max(lifetime) / count(features)
                    return float(np.max(lifetimes) / len(lifetimes))
        except Exception:
            pass

        return 0.0

    def is_hallucination(self, text, threshold=0.15):
        """
        Quick check if text is likely a hallucination

        Args:
            text: Text to verify
            threshold: R-score threshold

        Returns:
            bool: True if likely hallucination
        """
        result = self.analyze(text, threshold=threshold)
        return result['classification'] == 'hallucinated'


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("HEIMDALL - Usage Example")
    print("="*70)

    # Initialize detector
    detector = HeimdallDetector(
        model_name="gpt2",
        use_quantization=False
    )

    # Test cases
    test_cases = [
        ("The capital of France is Paris.", "Factual"),
        ("The capital of France is London.", "Hallucination"),
        ("Water boils at 100 degrees Celsius at sea level.", "Factual"),
        ("Water boils at 500 degrees Celsius at sea level.", "Hallucination"),
        ("Albert Einstein developed the theory of relativity.", "Factual"),
        ("Albert Einstein invented the internet in 1990.", "Hallucination")
    ]

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    for text, expected in test_cases:
        result = detector.analyze(text)

        status = "[!] HALLUCINATION" if result['classification'] == 'hallucinated' else "[OK] COHERENT"
        print(f"\n{status} (R={result['r_score']:.4f})")
        print(f"  Text: {text}")
        print(f"  Expected: {expected}")
        print(f"  Confidence: {result['confidence']:.2%}")

    print("\n" + "="*70)
    print("END OF EXAMPLE")
    print("="*70)

    # Integration example
    print("\n\n# Integration in your code:")
    print("""
    from heimdall import HeimdallDetector

    # Initialize once
    detector = HeimdallDetector(model_name="mistralai/Mistral-7B-v0.3", use_quantization=True)

    # Use multiple times
    def verify_llm_output(text):
        result = detector.analyze(text)
        if result['classification'] == 'hallucinated':
            return f"REJECT - Possible hallucination (confidence: {result['confidence']:.2%})"
        else:
            return f"ACCEPT - Coherent text (confidence: {result['confidence']:.2%})"

    # In production
    llm_output = "The Earth is flat and square."
    decision = verify_llm_output(llm_output)
    print(decision)
    """)
