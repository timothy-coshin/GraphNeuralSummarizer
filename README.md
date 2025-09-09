<div align="center">

# Graph Neural Summarizer (GNS)
**Addressing Information Bottlenecks in Graph-Augmented Large Language Models**


[![status](https://img.shields.io/badge/status-under%20review-blue.svg)](https://www.sciencedirect.com/journal/information-fusion)

</div>

> This repository contains the implementation for the manuscript **“Addressing Information Bottlenecks in Graph-Augmented Large Language Models via Graph Neural Summarization (GNS)”**, submitted to [Information Fusion](https://www.sciencedirect.com/journal/information-fusion).


---

## Code Release
- Code release is tied to the peer-review timeline. Interfaces are provided for easy adoption once public.

---

## ✨ At a Glance
- **Problem.** Graph-level prompting often compresses all node embeddings into a **single vector**, causing severe information loss.
- **Idea.** **GNS** introduces **query-aware multi-vector summarization** with a three-stage pipeline (**GNN_query → GNN_node → GNN_pool**) to preserve structural granularity.
- **Results.** On GraphQA include ExplaGraphs (commonsense QA), SceneGraphs (scene understanding), and WebQSP (KGQA), GNS shows strong performance and surpasses prior graph-prompting baselines.

---

## 📚 Table of Contents
- [Abstract](#abstract)
- [Key Contributions](#key-contributions)
- [Authors & Affiliation](#authors--affiliation)

---

## Abstract
> This study explores the challenge of integrating graph-structured data into large language models (LLMs) by addressing information bottlenecks in graph-level prompting. A key criticism is the severe information loss caused by compressing all node embeddings into a single vector. To address this, we propose the **Graph Neural Summarizer (GNS)**, a continuous prompting framework that mitigates this bottleneck via **query-aware multi-vector summarization**. The framework dynamically emphasizes query-relevant nodes and clusters semantically similar nodes for concise, context-aligned summaries.  
> Through extensive experiments on **ExplaGraphs**, **SceneGraphs**, and **WebQSP**, the GNS demonstrates superior performance, surpassing **G-Retriever**. These findings highlight GNS as a new paradigm for integrating graph-structured knowledge with LLMs.

---

## Key Contributions
- **Information Bottleneck Identification.** First explicit analysis of information loss in graph-level prompting for LLMs.  
- **GNS Framework.** A single-pass, three-stage GNN pipeline (**GNN_query → GNN_node → GNN_pool**) that yields **multi-token** prompts while preserving topology.  
- **Query-Aware Summarization.** Relevance-aware weighting + fixed-K clustering for compact, **structure-faithful** summaries scalable to large graphs.

---

## Authors & Affiliation
- **Wooyoung Kim**, Ph.D. (✉️ [timothy@yonsei.ac.kr](mailto:timothy@yonsei.ac.kr))  
- **Wooju Kim**, Ph.D. (✉️ [wkim@yonsei.ac.kr](mailto:wkim@yonsei.ac.kr))  
**Department of Industrial Engineering, Yonsei University**, Seoul, Republic of Korea

