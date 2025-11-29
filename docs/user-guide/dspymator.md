# DSPyMator

!!! note "Requires DSPy"
    This estimator requires the `dspy` optional dependency. Install with:
    ```bash
    uv add centimators[dspy]
    ```

`centimators.model_estimators.DSPyMator` brings the power of Large Language Models (LLMs) to feature engineering and tabular prediction tasks through [DSPy](https://dspy.ai/). Unlike traditional neural networks that learn patterns from data through gradient descent, DSPyMator leverages pre-trained LLMs and natural language reasoning to make predictions, making it uniquely suited for tasks where domain knowledge, explainability, and few-shot learning are critical.

## Why Use DSPyMator?

DSPyMator excels in scenarios where traditional machine learning falls short:

- **Few-Shot Learning**: Achieve strong performance with limited training data by leveraging the LLM's pre-existing knowledge
- **Domain Knowledge Integration**: Incorporate reasoning and expert knowledge naturally through task descriptions
- **Explainable Predictions**: Access the model's reasoning process (when using chain-of-thought)
- **Mixed Data Types**: Seamlessly handle numerical, categorical, and text features without complex preprocessing
- **Rapid Prototyping**: Get baseline predictions quickly before investing in traditional model training
- **Scikit-learn Compatible**: Stack and compose DSPyMator in scikit-learn pipelines, column transformers, and with other compatible workflows

## How It Works

DSPyMator wraps any DSPy `Module` (like `dspy.Predict` or `dspy.ChainOfThought`) and exposes it through the familiar scikit-learn API. Under the hood:

1. **Signature Definition**: You define input and output fields via DSPy signatures (e.g., `"review_text -> sentiment"`)
2. **Feature Mapping**: DSPyMator automatically maps your dataframe columns to the signature's input fields
3. **LLM Execution**: During prediction, each row is converted into a prompt and sent to the LLM
4. **Output Extraction**: Results are extracted and returned as numpy arrays (for `predict`) or dataframes (for `transform`)

The real power comes from DSPy's **optimization capabilities**. You can use optimizers like `GEPA`, `MIPROv2`, or `BootstrapFewShot` to automatically improve prompts, select better demonstrations, or even finetune the model—all through the standard `fit()` method.

## Usage

### Basic Classification

Let's start with a simple sentiment classification task using movie reviews:

```python
import polars as pl
import dspy
from centimators.model_estimators import DSPyMator

# Sample movie reviews
reviews = pl.DataFrame({
    "review_text": [
        "This movie was absolutely fantastic! A masterpiece.",
        "Terrible waste of time. Boring and predictable.",
        "Pretty good, though it had some slow moments.",
        "One of the worst films I've ever seen.",
        "Loved every minute of it! Highly recommended."
    ],
    "sentiment": ["positive", "negative", "neutral", "negative", "positive"]
})

# Define a DSPy program with input and output signature
# You can use dspy.Predict for simple predictions or dspy.ChainOfThought for reasoning
classifier_program = dspy.Predict("review_text: str -> sentiment: str")

# Create the DSPyMator estimator
sentiment_classifier = DSPyMator(
    program=classifier_program,
    target_names="sentiment",  # Which output field to use as predictions
    lm="openai/gpt-4o-mini",   # Language model to use
    temperature=0.0,            # Low temperature for consistent outputs
)

# Fit the classifier (establishes the LM configuration)
X = reviews[["review_text"]]
y = reviews["sentiment"]
sentiment_classifier.fit(X, y)

# Make predictions
test_reviews = pl.DataFrame({
    "review_text": [
        "An incredible journey with stunning visuals.",
        "Could barely stay awake through this one."
    ]
})

predictions = sentiment_classifier.predict(test_reviews[["review_text"]])
print(predictions)  # ['positive', 'negative']
```

### Getting Full Outputs with Transform

Unlike `predict()` which returns only the target field, `transform()` returns all output fields from the DSPy program:

```python
# Get all outputs (useful for accessing reasoning, confidence, etc.)
full_outputs = sentiment_classifier.transform(test_reviews[["review_text"]])
print(full_outputs)
# Polars DataFrame with column: sentiment
```

If you're using `dspy.ChainOfThought` instead of `dspy.Predict`, you'll also get reasoning:

```python
# Using ChainOfThought to get reasoning
cot_program = dspy.ChainOfThought("review_text: str -> sentiment: str")

cot_classifier = DSPyMator(
    program=cot_program,
    target_names="sentiment",
)

cot_classifier.fit(X, y)
outputs = cot_classifier.transform(test_reviews[["review_text"]])
print(outputs)
# Polars DataFrame with columns: rationale, sentiment
```

### Multi-Output Predictions

DSPyMator supports multiple output fields for richer predictions:

```python
# Define a program with multiple outputs
multi_output_program = dspy.Predict(
    "review_text: str -> sentiment: str, confidence: float"
)

multi_classifier = DSPyMator(
    program=multi_output_program,
    target_names=["sentiment", "confidence"],  # Multiple targets
)

multi_classifier.fit(X, y)
predictions = multi_classifier.predict(test_reviews[["review_text"]])
print(predictions.shape)  # (2, 2) - rows × outputs
```

### Prompt Optimization with GEPA

The real magic happens when you use DSPy optimizers to automatically improve your prompts:

```python
# Define a metric function for optimization
def sentiment_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Metric that returns score and optional feedback for GEPA.
    
    Args:
        gold: The ground truth example
        pred: The predicted output
        trace: Optional full program trace
        pred_name: Optional name of predictor being optimized
        pred_trace: Optional trace of specific predictor
    
    Returns:
        float score or dspy.Prediction(score=float, feedback=str)
    """
    y_pred = pred.sentiment
    y_true = gold.sentiment
    is_correct = (y_pred == y_true)
    score = 1.0 if is_correct else 0.0
    
    # If GEPA is requesting feedback, provide rich textual guidance
    if pred_name:
        if is_correct:
            feedback = f"Correctly classified as {y_pred}."
        else:
            feedback = f"Incorrect. Predicted {y_pred} but should be {y_true}."
        
        return dspy.Prediction(score=score, feedback=feedback)
    
    return score

# Create a GEPA optimizer
gepa_optimizer = dspy.GEPA(
    metric=sentiment_metric,
    auto="light",  # or "medium", "heavy" for more thoroughness
    reflection_minibatch_size=20,
    reflection_lm=dspy.LM(model="openai/gpt-4o-mini", temperature=1.0)
)

# Create a fresh classifier
optimized_classifier = DSPyMator(
    program=dspy.Predict("review_text: str -> sentiment: str"),
    target_names="sentiment",
)

# Fit with optimization (GEPA will improve the prompts)
optimized_classifier.fit(
    X, 
    y, 
    optimizer=gepa_optimizer,
    validation_data=0.3  # Use 30% of data for validation
)

# The optimized program is now ready to use
predictions = optimized_classifier.predict(test_reviews[["review_text"]])
```

### Few-Shot Learning with Bootstrap

For few-shot learning, use `BootstrapFewShot` to automatically select good demonstrations:

```python
# Few-shot optimizer doesn't need validation data
bootstrap_optimizer = dspy.BootstrapFewShot(
    metric=sentiment_metric,
    max_bootstrapped_demos=3,  # Number of examples to use
)

few_shot_classifier = DSPyMator(
    program=dspy.ChainOfThought("review_text: str -> sentiment: str"),
    target_names="sentiment",
)

# Fit with bootstrap (no validation_data needed)
few_shot_classifier.fit(
    X,
    y,
    optimizer=bootstrap_optimizer,
    validation_data=None  # Few-shot optimizers only need trainset
)

predictions = few_shot_classifier.predict(test_reviews[["review_text"]])
```

### Advanced: Custom Multi-Input Features

DSPyMator automatically maps multiple dataframe columns to signature fields:

```python
# Multi-feature example
movie_data = pl.DataFrame({
    "title": ["The Matrix", "Cats"],
    "review_text": ["Mind-bending sci-fi classic", "A catastrophic mistake"],
    "rating": [5, 1],
})

# Signature with multiple inputs
multi_input_program = dspy.Predict(
    "title: str, review_text: str, rating: int -> sentiment: str"
)

multi_input_classifier = DSPyMator(
    program=multi_input_program,
    target_names="sentiment",
    feature_names=["title", "review_text", "rating"],  # Map columns to signature
)

# Fit and predict
multi_input_classifier.fit(movie_data[["title", "review_text", "rating"]], None)
predictions = multi_input_classifier.predict(movie_data[["title", "review_text", "rating"]])
```

### Configuring Language Models

DSPyMator accepts either a model string or a pre-configured `dspy.LM` object for the `lm` parameter.

**Simple usage with model string:**

```python
# Uses default OpenAI API (requires OPENAI_API_KEY env var)
classifier = DSPyMator(
    program=dspy.Predict("text -> label"),
    target_names="label",
    lm="openai/gpt-4o-mini",
    temperature=0.0,
    max_tokens=1000,
)
```

**Using custom providers:**

For custom API configuration, pass a pre-configured `dspy.LM` object. DSPy uses [LiteLLM](https://docs.litellm.ai/) under the hood, so any LiteLLM-supported provider works. Extra kwargs are passed through to LiteLLM:

```python
import dspy

# Pass a pre-configured LM object
classifier = DSPyMator(
    program=dspy.Predict("text -> label"),
    target_names="label",
    lm=dspy.LM(
        "openrouter/anthropic/claude-3-haiku",
        temperature=0.1,
        max_tokens=1000,
        # Additional kwargs are passed to LiteLLM
    ),
)
```

!!! note "Temperature and max_tokens"
    When passing a `dspy.LM` object, configure `temperature` and `max_tokens` on the LM directly. The DSPyMator parameters are ignored when using a pre-configured LM.

**Environment variables:**

Most providers are configured via environment variables. Set them before calling `fit()`:

```python
import os

# OpenAI
os.environ["OPENAI_API_KEY"] = "sk-..."

# OpenRouter
os.environ["OPENROUTER_API_KEY"] = "sk-or-..."

# Anthropic
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

See the [DSPy LM documentation](https://dspy.ai/api/models/LM/) and [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) for supported providers and configuration.

### Async Execution for Speed

By default, DSPyMator uses async execution for faster batch predictions:

```python
# Async is on by default
fast_classifier = DSPyMator(
    program=dspy.Predict("review_text: str -> sentiment: str"),
    target_names="sentiment",
    use_async=True,        # Default behavior
    max_concurrent=50,     # Max concurrent API requests
    verbose=True,          # Show progress bar
)

# For synchronous execution (useful for debugging)
sync_classifier = DSPyMator(
    program=dspy.Predict("review_text: str -> sentiment: str"),
    target_names="sentiment",
    use_async=False,
    verbose=True,
)
```

!!! warning "AsyncIO and Local Model Limitations"
    DSPyMator's async execution mode is ideal for rapid API-backed LLM calls (e.g., OpenAI, Anthropic) because it uses asyncio. For best results with local LLMs, use `use_async=False` unless you know your backend supports true parallelism.


## Pipeline Integration

DSPyMator works seamlessly in scikit-learn pipelines:

```python
from sklearn.pipeline import make_pipeline

llm_pipeline = make_pipeline(
    # ... preprocessing steps ...
    DSPyMator(
        program=dspy.ChainOfThought("review_text: str -> sentiment: str"),
        target_names="sentiment"
    ),
    EmbeddingTransformer(
        model="openai/text-embedding-3-small",      # or your preferred embedding model
        feature_names=["reasoning"],                # specify which columns to embed
    )
```

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `program` | `dspy.Module` | *required* | DSPy module (e.g., `dspy.Predict`, `dspy.ChainOfThought`) with a signature defining input/output fields |
| `target_names` | `str \| list[str]` | *required* | Output field name(s) to use as predictions |
| `feature_names` | `list[str] \| None` | `None` | Column names mapping input data to signature fields. If `None`, inferred from dataframe columns |
| `lm` | `str \| dspy.LM` | `"openai/gpt-5-nano"` | Language model - either a string identifier or a pre-configured `dspy.LM` object |
| `temperature` | `float` | `1.0` | Sampling temperature (ignored if `lm` is a `dspy.LM` object) |
| `max_tokens` | `int` | `16000` | Maximum tokens in responses (ignored if `lm` is a `dspy.LM` object) |
| `use_async` | `bool` | `True` | Use async execution for batch predictions |
| `max_concurrent` | `int` | `50` | Maximum concurrent requests in async mode |
| `verbose` | `bool` | `True` | Show progress bars during prediction |

### fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | DataFrame/array | *required* | Training features |
| `y` | Series/array | *required* | Target values (can be `None` for unsupervised) |
| `optimizer` | `dspy.Optimizer \| None` | `None` | DSPy optimizer instance (e.g., `dspy.GEPA`, `dspy.BootstrapFewShot`) |
| `validation_data` | `tuple \| float \| None` | `None` | Validation data as `(X_val, y_val)`, a float for train split fraction, or `None` |