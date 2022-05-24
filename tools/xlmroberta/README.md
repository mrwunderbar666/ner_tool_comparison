# Info

Using the huggingface library for quick implementation.

# Hyperparameters

Several sweeps for hyperparamter optimization yielded the following optimal settings for XLM-ROBERTa:

- Learning Rate: 2e-05
- Batch Size: 8
- Epochs: 4
- Weight Decay: 0.01


## Results

Note: Tuned on the CoNLL++ Dataset.

- accuracy 0.99101
- f1 0.94553
- precision 0.94026
- recall 0.95086
- loss 0.03841
- runtime 3.6195 minutes
- samples_per_second 897.912
- steps_per_second 28.181
- train/epoch 4.0
