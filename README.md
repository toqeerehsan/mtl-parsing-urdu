README.md: |
  # Urdu Parsing Toolkit: Multi-Task Constituency and Dependency Parsing

  [![Paper](https://img.shields.io/badge/Paper-PLOS%20ONE%202025-blue)](https://doi.org/10.1371/journal.pone.0332580)
  [![Language](https://img.shields.io/badge/Language-Urdu-green)]()
  [![Task](https://img.shields.io/badge/Tasks-Constituency%20Parsing%20%7C%20Dependency%20Parsing-orange)]()
  [![NLP](https://img.shields.io/badge/Area-Low--Resource%20NLP-purple)]()
  [![License](https://img.shields.io/badge/License-Research-lightgrey)]()

  
  Official repository for the PLOS ONE 2025 paper:

  **Multi-task learning by using contextualized word representations for syntactic parsing of a morphologically rich language**

  **Authors:** Toqeer Ehsan, Miriam Butt, Sarmad Hussain, Hassan Alhuzali, Ali Al-Laith

  **Paper:** https://doi.org/10.1371/journal.pone.0332580

  **Repository:** https://github.com/toqeerehsan/mtl-parsing-urdu

  ---

  ## Overview

  This repository provides code, data, models, and conversion utilities for Urdu syntactic parsing.

  The work focuses on **Urdu**, a morphologically rich and low-resource language, and introduces a multi-task learning framework for jointly learning:

  - Constituency Parsing
  - Dependency Parsing

  The system uses sequence labeling, contextualized Urdu word representations, and shared neural representations to improve parsing performance.

  ---

  ## Why This Repository Is Useful

  This repository is useful for researchers working on:

  - Urdu NLP
  - Low-resource NLP
  - Dependency parsing
  - Constituency parsing
  - Treebank conversion
  - Universal Dependencies
  - Multi-task learning
  - Contextualized embeddings
  - Sequence labeling for parsing
  - Morphologically rich languages

  The resources can support research in parsing, information extraction, machine translation, question answering, semantic parsing, and large language model evaluation for low-resource languages.

  ---

  ## Main Contributions

  ### 1. Urdu Dependency Treebank Conversion

  This work converts the CLE Urdu Treebank from phrase structure representation into dependency structure using:

  - Language-specific head-word rules
  - Phrase-to-dependency label mappings
  - Universal Dependencies compatible labels
  - Post-conversion correction rules

  ### 2. Sequence Labeling for Parsing

  Both constituency and dependency parsing are reformulated as sequence labeling tasks.

  This makes parsing efficient and allows the use of a unified neural architecture.

  ### 3. Contextualized Urdu Word Representations

  Contextualized Urdu word representations are trained on a large Urdu corpus of approximately:

  **220 million tokens**

  ### 4. Multi-Task Learning Framework

  The proposed framework jointly learns constituency and dependency parsing.

  The multi-task setup improves generalization by learning shared syntactic representations across both parsing tasks.

  ---

  ## Main Results

  ### Constituency Parsing

  | Model | F1 Score |
  |---|---:|
  | Previous state of the art | 88.10 |
  | Proposed multi-task parser | **91.39** |

  **Improvement:** +3.29 F1

  ### Dependency Parsing

  | Model | LAS |
  |---|---:|
  | Previous state of the art | 84.20 |
  | Proposed multi-task parser | **85.69** |

  **Improvement:** +1.49 LAS

  ---

  ## Resources Included

  This repository includes:

  - CLE-UTB constituency parsing resources
  - CLE-UTB to dependency conversion scripts
  - Universal Dependencies label mappings
  - Sequence labeling representations
  - Multi-task parsing code
  - Contextualized Urdu embedding support
  - Evaluation scripts
  - Tree conversion utilities
  - Reproducible experiment files

## Citation

  If you use this repository, code, models, embeddings, dataset, conversion rules, or results, please cite the following paper:

  ```bibtex
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
  ```

  ## Related Publications

  Please also consider citing the related Urdu treebank and dependency parsing resources:

  ```bibtex
  title={Development and evaluation of an Urdu treebank (CLE-UTB) and a statistical parser},
  author={Ehsan, Toqeer and Hussain, Sarmad},
  journal={Language Resources and Evaluation},
  volume={55},
  number={2},
  pages={287--326},
  year={2021},
  publisher={Springer}
  }
  ```

  ```bibtex
  title={Dependency parsing for Urdu: Resources, conversions and learning},
  author={Ehsan, Toqeer and Butt, Miriam},
  booktitle={Proceedings of the Twelfth Language Resources and Evaluation Conference},
  pages={5202--5207},
  year={2020}
  }  
  ```
  
  ```bibtex
  @article{ehsan2019analysis,
  title={Analysis of experiments on statistical and neural parsing for a orphologically rich and free word order language Urdu},
  author={Ehsan, Toqeer and Hussain, Sarmad},
  journal={IEEE Access},
  volume={7},
  pages={161776--161793},
  year={2019},
  publisher={IEEE}
  }
  ```
