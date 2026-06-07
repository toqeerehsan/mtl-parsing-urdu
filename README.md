# Multi-Task Learning by using Contextualized Word Representations for Syntactic Parsing of a Morphologically Rich Language

This repository contains code, models, and data for the paper:

**Multi-Task Learning by using Contextualized Word Representations for Syntactic Parsing of a Morphologically-rich Language**  
*Toqeer Ehsan, Miriam Butt, Sarmad Hussain, Hassan Alhuzali and Ali Al-Laith*   
Published in *PLOS ONE*, 2025

---

## Overview

We propose a **multi-task learning (MTL)** framework that jointly trains a constituency parser and a semantic dependency parser for **low-resource languages**, with a particular focus on **Urdu**.

The system leverages shared representations to benefit from syntactic-semantic correlations, resulting in improved generalization and performance in both tasks.

This is the **first publicly available dataset and parser for joint syntactic-semantic analysis in Urdu**.

---

## Key Features

- Joint learning of constituency and semantic parsing  
- Uses gold-standard and weakly labeled corpora for Urdu  
- Model variants with ELMo, GloVe, and Word2Vec embeddings  
- Evaluation on labeled F1-score and semantic parsing accuracy  
- Supports multitask optimization with hard and soft parameter sharing  

---

## 📂 Project Structure
```yaml
.
├── data/ # Urdu syntactic & semantic datasets
├── mtl_parser.py # Main multitask parser code
├── labels2brackets.py # Utility for converting to bracketed format
├── tree2.py # Semantic tree conversion script
├── .gitignore # Files/folders to be ignored by Git
└── .gitattributes # Git LFS tracked files
```

---

## Dataset

The datasets are based on:

- A syntactic corpus built from Urdu Treebank-style annotations  
- A grammatical role labeling structure using Urdu PropBank-style structures  

---

## Requirements

- Python 3.7+  
- TensorFlow or Keras (for model training)  
- NLTK  
- NumPy, SciPy, and other common NLP libraries  

```bash
# Install all dependencies:

pip install -r requirements.txt

# Running the Parser
python mtl_parser.py

# Evaluation
python labels2brackets.py      # Converts predicted parse trees  
python tree2.py                # Converts semantic parses  
```

---

## Citations
if you use this code or dataset, please cite:

```yaml
@article{ehsan2025multi,
  title={Multi-task learning by using contextualized word representations for syntactic parsing of a morphologically rich language},
  author={Ehsan, Toqeer and Butt, Miriam and Hussain, Sarmad and Alhuzali, Hassan and Al-Laith, Ali},
  journal={Plos one},
  volume={20},
  number={9},
  pages={e0332580},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}

@article{ehsan2021development,
  title={Development and evaluation of an Urdu treebank (CLE-UTB) and a statistical parser},
  author={Ehsan, Toqeer and Hussain, Sarmad},
  journal={Language Resources and Evaluation},
  volume={55},
  number={2},
  pages={287--326},
  year={2021},
  publisher={Springer}
}

@inproceedings{ehsan2020dependency,
  title={Dependency parsing for Urdu: Resources, conversions and learning},
  author={Ehsan, Toqeer and Butt, Miriam},
  booktitle={Proceedings of the Twelfth Language Resources and Evaluation Conference},
  pages={5202--5207},
  year={2020}
}

@article{ehsan2019analysis,
  title={Analysis of experiments on statistical and neural parsing for a morphologically rich and free word order language Urdu},
  author={Ehsan, Toqeer and Hussain, Sarmad},
  journal={IEEE Access},
  volume={7},
  pages={161776--161793},
  year={2019},
  publisher={IEEE}
}
```
