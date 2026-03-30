# Graph Neural Summarizer (GNS)

Official implementation of:

> **Addressing Information Bottlenecks in Graph Augmented Large Language Models via Graph Neural Summarization**  
> Wooyoung Kim, Wooju Kim  
> *Information Fusion*, 2026  
> [Paper](https://doi.org/10.1016/j.inffus.2025.103784)

## Overview

GNS is a continuous prompting framework that mitigates information bottlenecks in graph-level prompting by generating multiple query-aware prompt vectors. Instead of compressing all node embeddings into a single vector, GNS clusters nodes and summarizes each cluster through a dedicated learnable token, preserving richer structural information for LLM reasoning.

## Setup

This codebase extends [G-Retriever](https://github.com/XiaoxinHe/G-Retriever) (He et al., NeurIPS 2024). We provide only the files we added or modified; all other components (datasets, utilities, base models) come from G-Retriever.

### 1. Clone G-Retriever (pinned commit)

```bash
git clone https://github.com/XiaoxinHe/G-Retriever.git
cd G-Retriever
git checkout <COMMIT_HASH>   # TODO: pin to the commit you used
```

### 2. Install dependencies

Follow G-Retriever's environment setup, then install the additional dependency:

```bash
conda create --name gns python=3.9 -y
conda activate gns

# PyTorch (adjust CUDA version as needed)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# PyG
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Other dependencies
pip install peft pandas ogb transformers wandb sentencepiece scikit-learn
```

### 3. Copy GNS files

```bash
# From this repository root:
bash setup.sh /path/to/G-Retriever
```

Or manually:

```bash
# Model files → G-Retriever/src/model/
cp src/model/gnn.py        /path/to/G-Retriever/src/model/gnn.py
cp src/model/gns_llm.py    /path/to/G-Retriever/src/model/gns_llm.py
cp src/model/__init__.py    /path/to/G-Retriever/src/model/__init__.py

# Config → G-Retriever/src/
cp src/config.py            /path/to/G-Retriever/src/config.py

# Training script → G-Retriever/
cp train_gns.py             /path/to/G-Retriever/train_gns.py
```

### 4. Prepare query embeddings

GNS uses pre-computed query embeddings from Sentence Transformer. Generate them before training:

```bash
cd /path/to/G-Retriever
python preprocess_query_embeddings.py --dataset expla_graphs
python preprocess_query_embeddings.py --dataset scene_graphs
python preprocess_query_embeddings.py --dataset webqsp
```

This saves `q_embs.pt` under `dataset/{dataset_name}/`.

> **Note**: If `preprocess_query_embeddings.py` is not included in this repo, you can generate `q_embs.pt` with:
> ```python
> from sentence_transformers import SentenceTransformer
> model = SentenceTransformer('all-roberta-large-v1')
> # Encode all questions in the dataset and save as torch tensor
> ```

## Training

```bash
cd /path/to/G-Retriever

# ExplaGraphs
python train_gns.py --dataset expla_graphs --model_name gns_llm --num_graph_token 8 --seed 0

# SceneGraphs
python train_gns.py --dataset scene_graphs --model_name gns_llm --num_graph_token 8 --seed 0

# WebQSP
python train_gns.py --dataset webqsp --model_name gns_llm --num_graph_token 8 --seed 0
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `graph_llm` | Set to `gns_llm` for GNS |
| `--num_graph_token` | `8` | Number of prompt vectors (k) |
| `--query_aware` | `True` | Enable query-aware encoding |
| `--pooling` | `graph_token` | `graph_token` (GNS) / `mean` / `sum` |
| `--gnn_model_name` | `gt` | GNN backbone: `gt` / `gat` / `gcn` |
| `--num_epochs` | `10` | Training epochs |
| `--llm_frozen` | `True` | Freeze LLM (set `False` for LoRA) |

## Results

Performance on GraphQA benchmarks (Frozen LLM):

| Method | # Prompt Tokens | ExplaGraphs | SceneGraphs | WebQSP |
|---|---|---|---|---|
| G-Retriever | 1 | 85.16 | 81.31 | 70.49 |
| **GNS (Ours)** | **8** | **92.33** | **84.68** | **73.28** |

## Repository Structure

```
GraphNeuralSummarizer/
├── README.md
├── setup.sh                  # Automated setup script
├── train_gns.py              # Training script
└── src/
    ├── config.py             # Argument parser with GNS-specific args
    └── model/
        ├── __init__.py       # Model registry (adds gns_llm)
        ├── gnn.py            # GNN backbones + GNS encoder
        └── gns_llm.py        # GNS-LLM integration module
```

## Citation

```bibtex
@article@article{KIM2026103784,
title = {Addressing information bottlenecks in graph augmented large language models via graph neural summarization},
journal = {Information Fusion},
volume = {127},
pages = {103784},
year = {2026},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.103784},
url = {https://www.sciencedirect.com/science/article/pii/S1566253525008462},
author = {Wooyoung Kim and Wooju Kim},
keywords = {Graph neural network, Large language model, Continuous prompting, Information bottleneck, Graph question answering},
abstract = {This study investigates the problem of information bottlenecks in graph-level prompting, where compressing all node embeddings into a single vector leads to significant structural information loss. We clarify and systematically analyze this challenge, and propose the Graph Neural Summarizer (GNS), a continuous prompting framework that generates multiple query-aware prompt vectors to better preserve graph structure and improve context relevance. Experiments on ExplaGraphs, SceneGraphs, and WebQSP show that GNS consistently improves performance over strong graph-level prompting baselines. These findings emphasize the importance of addressing information bottlenecks when integrating graph-structured data with large language models. Implementation details and source code are publicly available at https://github.com/timothy-coshin/GraphNeuralSummarizer.}
}
```

## Acknowledgments

This codebase builds upon [G-Retriever](https://github.com/XiaoxinHe/G-Retriever) by He et al. We thank the authors for making their code publicly available.
