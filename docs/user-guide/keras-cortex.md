# KerasCortex

!!! Warning
    This module is a work in progress. It is not yet ready for production use.
    This is highly experimental and likely to overfit.

`centimators.model_estimators.KerasCortex` introduces a novel approach to model development by automating aspects of architecture search. It wraps a Keras-based estimator and leverages a Large Language Model (LLM) to recursively self-reflect on its own architecture. The LLM suggests improvements to the model's source code, which are then tested. This iterative process allows `KerasCortex` to refine its internal model over several cycles, potentially discovering more optimal architectures for the given data.

## How It Works

At its core, `KerasCortex` utilizes DSPy to manage the interaction with the LLM through two key components:

**The `Think` Module**: A DSPy `Module` that orchestrates the LLM's code generation process. It uses DSPy's `ChainOfThought` to enable access to the LLM's reasoning process as it improves its own architecture. The `Think` module takes the current `build_model` source code, a history of attempted code modifications and their performance, and an optimization goal (e.g., "improve validation R2 score"), then returns the LLM's suggested `build_model` method modification.

**The `think_loop` Method**: The heart of `KerasCortex`'s self-improvement mechanism. This iterative process works as follows:

1. **Establish Baseline**: The initial Keras estimator is cloned and trained on the training data to establish baseline performance on the validation set
2. **Refine Architecture**: For each iteration:
    - The `Think` module suggests a new `build_model` code modification based on the current best code and performance history of all previous iterations
    - The suggested code is executed to create a new `build_model` function
    - A new model instance is created with the modified architecture and trained on the data
    - The model is evaluated on validation data and compared to the current best
    - If performance improves, the new model becomes the best candidate; each model and its performance is logged for future iterations to reflect upon
3. **Converge?**: After all iterations (or early termination due to errors), the best-performing model and complete performance log are returned

`KerasCortex` requires validation data to evaluate the performance of different architectures. It uses this information to guide its self-improvement process. The `lm` parameter specifies the language model to be used for code generation, and `n_iterations` controls how many times the model attempts to refine itself. When `verbose=True`, you can observe the LLM's reasoning process and see how it decides to modify the architecture at each step. 

![KerasCortex Reasoning](../assets/images/reasoning.png)

This approach allows `KerasCortex` to explore different architectural modifications and converge towards a model that performs well on the given validation data. One could even finetune the prompts or LLM weights directly to improve the quality of the suggestions in its own meta-loop. Access to tools like the keras documentation or arxiv papers could be added as well.



## Usage

While `KerasCortex` is an advanced tool, its scikit-learn compatible API makes it surprisingly easy to integrate into existing workflows.

```python
# Define a base Keras estimator, can be anything with a `build_model` method
base_mlp = MLPRegressor(
    hidden_units=(64, 32),
    dropout_rate=0.1,
)

# Initialize KerasCortex
# Ensure your LM (e.g., OpenAI API key) is configured in your environment
cortex = KerasCortex(
    base_estimator=base_mlp,
    n_iterations=5,  # Number of self-reflection iterations
    lm="openai/gpt-4o-mini", # Or any other dspy.LM compatible model
    verbose=True
)

# Fit KerasCortex like any scikit-learn estimator
cortex.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val), # Needed for self-reflection
    epochs=5, # Epochs for each iteration's model training
    batch_size=128,
)

# Make predictions with the best model found
predictions = cortex.predict(X_val)
```

View the [KerasCortex tutorial](../tutorials/keras-cortex.ipynb) for a more detailed example.