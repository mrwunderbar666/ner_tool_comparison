# XLM-RoBERTa

- Citation: Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2019). Unsupervised Cross-lingual Representation Learning at Scale. https://arxiv.org/abs/1911.02116
- Model Repository: https://huggingface.co/xlm-roberta-large

*Using the huggingface library for quick implementation.*

*Using [wandb](https://wandb.ai/) for automation controls*

# Hyperparameters

Several sweeps for hyperparameters optimization yielded the following optimal settings for XLM-ROBERTa:

- Learning Rate: 2e-05
- Batch Size: 8
- Epochs: 4

You can replicate the search for hyperparameters by using the script `find_hyperparameters.py`. The parameters to search for are notes in `find_hyperparameters.yml`. Script is written for the service: [wandb](https://wandb.ai/). You can run a sweep and use the yaml file for the sweep settings. Please note that this process can take a while (one run about 60 minutes).

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

# Fine-tuning

## Determine Combinations

We prepared the script `train_combinations.py` (alongside `combination_sweeps.yaml`). Script is written for the service: [wandb](https://wandb.ai/). You can run a sweep and use the yaml file for the sweep settings.

First time running the script, it automatically generates all possible combinations of languages and corpora (see function inside `tools/xlmroberta/utils.py`). You can call the script with the argument `-m 123` to train with the combination number 123. It will then save the model and also its parameters.

Note that this process takes a long time, especially when wikiann corpus is included.

## Use only subsets of training data

The scripts `train_varying_data_size.py` and `train_varying_single_corpus.py` are to train models with only a subset of the training data. We use this to determine, how "little" data is necessary to get acceptable model performance.

# Evaluation

Main script is `evaluate.py` which takes the model with the full training data and evaluates it against all validation datasets. This produces the metrics reported in the paper. 

The other evaluation scripts are for analyzing the impact of hyperparameters, corpora, and languages.