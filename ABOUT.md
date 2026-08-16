# About BBQ-RAG: Origins, Lineage & Acknowledgments

## Project Origins & Conceptual Lineage

BBQ-RAG was developed to explore high-throughput, low-latency late-interaction (MaxSim) visual document retrieval on standard CPU architectures. The core conceptual foundations of this project are directly built upon the pioneering work of the **ColPali** and **maxsim-cpu** communities.

---

## Foundational References & Contributions

### 1. ColPali: Efficient Document Retrieval with Vision Language Models

The visual embedding model architecture, multi-vector late-interaction paradigm for document images, and patch-level visual prompt tokenization in this project are based on the **ColPali** framework.

- **Research Paper**: [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)
- **Authors**: Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo
- **Official Repository**: [illuin-tech/colpali](https://github.com/illuin-tech/colpali)
- **Model Weights & Adapters**: [vidore on Hugging Face](https://huggingface.co/vidore)

### 2. maxsim-cpu: CPU Late-Interaction Scoring

The approach to accelerating multi-vector MaxSim scoring on CPU architectures and benchmarking CPU-based similarity operators was inspired by and designed in view of **maxsim-cpu**.

- **Package Reference**: [maxsim-cpu on PyPI](https://pypi.org/project/maxsim-cpu/)

---

## Acknowledgments & Appreciation

We express our sincere gratitude and appreciation to:

- The **ColPali Research Team** at Illuin Technology, CentraleSupélec, and Université Paris-Saclay for introducing the ColPali architecture and open-sourcing the `colpali_engine` codebase and ViDoRe benchmark suite.
- The **maxsim-cpu Maintainers** for their valuable contributions toward accelerating late-interaction operators on CPU systems.
- The broader open-source vision-language and retrieval research community.

---

## Rights & Intellectual Property Notice

All rights, titles, copyrights, and intellectual property in the original ColPali architecture, research papers, model architectures, weights, and the `colpali_engine` codebase remain strictly reserved to their respective authors, Illuin Technology, and affiliated research institutions.

All rights to `maxsim-cpu` belong to its respective creators and contributors.

This project is an independent open-source implementation and derivative exploration intended for research and educational purposes under the [Apache 2.0 License](LICENSE).
