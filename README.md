# Implicit Language Models (Standalone)

A standalone fork of [implicit_languagemodels](https://github.com/microsoft/implicit_languagemodels) with a self-contained fixed-point solver, removing the external [TorchDEQ](https://github.com/locuslab/torchdeq) dependency.

Based on the paper [Implicit Language Models are RNNs: Balancing Parallelization and Expressivity](https://arxiv.org/abs/2502.07827) (ICML 2025 Spotlight).

## What's Different

- **Standalone fixed-point solver** with phantom gradient backward pass, replacing TorchDEQ (`implicit_llm/solver.py`)
- **Simplified config dataclasses** — removed TorchDEQ-specific fields (separate forward/backward solvers, norm config, etc.)
- **Refactored variational dropout** into a single `VariationalDropout1d` class
- **Single-command install** — torch added as build-system requirement

# Installation
```
pip install .
```

# Usage
The code allows for integration with the HuggingFace Platform.
We provide local configuration files that can be loaded with `AutoConfig`
```python
from transformers import AutoConfig, AutoModel
import implicit_llm

cfg = AutoConfig.from_pretrained('hf_models/llama3-1.3b-implicit')
model = AutoModel.from_config(cfg)
```

# Examples

## State-Tracking
We provide a simple training script based on the huggingface Trainer. 
First, generate the dataset [following the instructions](state_tracking/README.md).
Then, train your models with
```python
python -m examples.state_tracking \ 
    --model_name hf_models/mamba2-state-tracking-implicit \
    --train_dataset /path/to/data/train_A5_L256_P090.bin \
    --eval_dataset /path/to/data/test_A5_L256_P050.bin \
    --test_dataset /path/to/data/test_A5_L256_P050.bin 
```
The script works for arbitrary models from the huggingface hub.
Feel free to train your favorite models!

To evaluate a trained model use the `--eval` flag and point `--model_name` to the trained model checkpoint. 
E.g. run evaluation on the test set with 1024 tokens
```python
python -m examples.state_tracking \ 
    --model_name path/to/trained/model/checkpoint \
    --train_dataset /path/to/data/train_A5_L256_P090.bin \
    --eval_dataset /path/to/data/test_A5_L256_P050.bin \
    --test_dataset /path/to/data/test_A5_L256_P050.bin
    --eval 
```

## Downstream Evaluation of Pretrained Models

## Duality of Simultaneous Fixed Point Iteration and Sequential Fixed Point Iteration
By default, training always used the simultaneous fixed point iteration, while generation always uses the sequential fixed point iteration.
We provide examples of evaluating a model in the sequential mode, e.g. to reproduce Figure 2C, in `tests/test_evaluation.py` and in `examples/state_tracking.py`.
The state tracking example code uses the simultaneous mode for validation during training.
A sequential pass is done at the end of training on the test set. 

# Common Issues
    ValueError: The checkpoint you are trying to load has model type `implicit_mamba2` but Transformers does not recognize this architecture.

--> Just `import implicit_llm` to register the implicit models with the HF library, or `
```
from implicit_llm import register_implicit_causal_lm
register_implicit_causal_lm()
```
# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

# Citation
```
@inproceedings{
schone2025implicit,
title={Implicit Language Models are {RNN}s: Balancing Parallelization and Expressivity},
author={Mark Sch{\"o}ne and Babak Rahmani and Heiner Kremer and Fabian Falck and Hitesh Ballani and Jannes Gladrow},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=5EbiopWH6e}
}
```