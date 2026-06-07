# Multi-Task Learning by using Contextualized Word Representations for Syntactic Parsing of a Morphologically Rich Language

**Multi-Task** We address the challenge of syntactic parsing for Urdu, a morphologically rich language, and present state-of-the-art results for both constituency and dependency parsing. This paper offers four major contributions: 1) the conversion of the CLE-UTB phrase structure treebank into a dependency treebank by developing language-specific head-word and phrase-to-dependency label mapping rules; 2) a novel sequence labeling scheme that transforms the parsing task into a unified representation; 3) the training of contextualized word representations on a large 220 million tokens Urdu corpus collected from the web; and 4) development of parsing framework using two learning paradigms, single-task and multi-task learning. Several post-processing rules are applied to improve the quality of the automatically converted dependency structure treebank. The proposed sequence labeling scheme enables the use of a shared architecture that learns the syntactic structures from both grammatical structures simultaneously and hence improves generalization. Experiments show that the multi-task learning setup significantly enhances parsing performance, achieving an F1 score of 91.39 for constituency parsing (an improvement of 3.29 points) and a labeled attachment score of 85.69 for dependency parsing (an improvement of 1.49 points). These results demonstrate that learning cross-task representations provides measurable benefits and advances the state of syntactic parsing for Urdu.

**Multi-Task Learning by using Contextualized Word Representations for Syntactic Parsing of a Morphologically-rich Language**  
*Toqeer Ehsan, Miriam Butt, Sarmad Hussain, Hassan Alhuzali and Ali Al-Laith*   
Published in *PLOS ONE*, 2025
This repository contains code, models, and data for the paper:

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
