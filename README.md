# Large Language Models on Tabular Data -- A Survey


```
@article{
fang2024large,
title={Large Language Models ({LLM}s) on Tabular Data: Prediction, Generation, and Understanding - A Survey},
author={Xi Fang and Weijie Xu and Fiona Anting Tan and Ziqing Hu and Jiani Zhang and Yanjun Qi and Srinivasan H. Sengamedu and Christos Faloutsos},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=IZnrCGF9WI},
note={}
}

```
[Original paper](https://arxiv.org/abs/2402.17944) 

# LLM on Tabular Data Prediction and Understanding -- A Survey
This repo is constructed for collecting and categorizing papers about diffusion models according to our survey paper——[_**Large Language Models on Tabular Data -- A Survey**_](https://arxiv.org/abs/2402.17944). Considering the fast development of this field, we will continue to update **both [arxiv paper](https://arxiv.org/abs/2402.17944) and this repo**.

**Abstract** \
Recent breakthroughs in large language modeling have facilitated rigorous exploration of their application in diverse tasks related to tabular data modeling, such as prediction, tabular data synthesis, question answering, and table understanding. Each task presents unique challenges and opportunities. However, there is currently a lack of comprehensive review that summarizes and compares the key techniques, metrics, datasets, models, and optimization approaches in this research domain. This survey aims to address this gap by consolidating recent progress in these areas, offering a thorough  survey and taxonomy of the datasets, metrics, and methodologies utilized. It identifies strengths, limitations, unexplored territories, and gaps in the existing literature, while providing some insights for future research directions in this vital and rapidly evolving field. It also provides relevant code and datasets references. Through this comprehensive review, we hope to provide interested readers with pertinent references and insightful perspectives, empowering them with the necessary tools and knowledge to effectively navigate and address the prevailing challenges in the field.


![336529724-fdd847f0-f232-474c-aaac-bc8232a42547](https://github.com/tanfiona/LLM-on-Tabular-Data-Prediction-Table-Understanding-Data-Generation/assets/56927837/60041a9c-ee76-4db5-9ccb-0d0e7880d322)
Figure 1: Overview of LLM on Tabular Data: the paper discusses application of LLM for prediction, data
generation, and table understanding tasks.


![LLMs_x_TabularData_KeyTechniques](https://github.com/tanfiona/LLM-on-Tabular-Data-Prediction-Table-Understanding-Data-Generation/assets/56927837/53f06913-ae85-4815-8039-973d886fc063)
Figure 4: Key techniques in using LLMs for tabular data. The dotted line indicates steps that are optional.


**Table of content:**
- [Taxonomy](#taxonomy)
  - [Prediction task](#prediction)
  - [Generation task](#generation)
  - [Table understanding task](#understanding)
- [Datasets](#datasets)
  - [Prediction task](#prediction-tasks)
  - [Generation task](#dgeneration)
  - [Table understanding task](#table-understanding-tasks)


<!-- headings -->
<a id="taxonomy"></a>
## Taxonomy

<a id="prediction"></a>


### Prediction task
---
#### Tabular Data
**[TABLET: Learning From Instructions For Tabular Data](https://arxiv.org/abs/2304.13188)**
**[[code](https://github.com/dylan-slack/Tablet)]**




**[Language models are weak learners](https://arxiv.org/abs/2306.14101)**   




**[LIFT: Language-Interfaced Fine-Tuning for Non-Language Machine Learning Tasks](https://arxiv.org/abs/2206.06565)**  
**[[code](https://github.com/UW-Madison-Lee-Lab/LanguageInterfacedFineTuning)]**



**[TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/abs/2210.10723)**   
**[[code](https://github.com/clinicalml/TabLLM)]**



**[UniPredict: Large Language Models are Universal Tabular Classifiers](https://arxiv.org/abs/2310.03266)**   




**[Towards Foundation Models for Learning on Tabular Data](https://arxiv.org/abs/2310.07338)**    
 



**[Towards Better Serialization of Tabular Data for Few-shot Classification with Large Language Models
](https://arxiv.org/abs/2312.12464)**  

**[Multimodal clinical pseudo-notes for emergency department prediction tasks using multiple embedding model for ehr (meme)](https://arxiv.org/abs/2402.00160)**
**[[code](https://github.com/Simonlee711/MEME)]

**[Text Serialization and Their Relationship with the Conventional Paradigms of Tabular Machine Learning](https://arxiv.org/abs/2406.13846)**

**[StructLM: Towards Building Generalist Models for Structured
Knowledge Grounding
](https://arxiv.org/pdf/2402.16671)**

**[UniTabE: A Universal Pretraining Protocol for Tabular Foundation Model in Data Science
](https://arxiv.org/abs/2307.09249)**  



**[Unleashing the Potential of Large Language Models for Predictive Tabular Tasks in Data Science
](https://arxiv.org/abs/2403.20208)**
**[[model](https://huggingface.co/OldBirdAZ/itab-llm)]**  

**[Synthetic Oversampling: Theory and A Practical Approach Using LLMs to Address Data Imbalance
](https://arxiv.org/abs/2406.03628)**  


#### Time series 

**[LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law](https://arxiv.org/abs/2402.00795)**

**[PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting](https://arxiv.org/abs/2210.08964)**   

**[Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2310.07820)**  
**[TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series](https://arxiv.org/abs/2308.08241)**  


**[Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)**  
**[[code](https://github.com/ngruver/llmtime)]**

#### Application Specific 

**[MediTab: Scaling Medical Tabular Data Predictors via Data Consolidation, Enrichment, and Refinement](https://arxiv.org/abs/2305.12081)**  
**[[code](https://github.com/RyanWangZf/MediTab)]**

**[CPLLM: Clinical Prediction with Large Language Models](https://arxiv.org/abs/2309.11295)**  
**[[code](https://github.com/nadavlab/CPLLM)]**

**[SERVAL : Synergy Learning between Vertical Models and LLMs towards Oracle-Level Zero-shot Medical Prediction](https://arxiv.org/pdf/2403.01570)**  



**[CTRL: Connect Collaborative and Language Model for CTR Prediction](https://arxiv.org/abs/2306.02841)**  




**[FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031)**  
**[[code](https://github.com/YuweiYin/FinPT)]**


<a id="generation"></a>
### Data Generation task
---
**[Language Models are Realistic Tabular Data Generators](https://arxiv.org/abs/2210.06280)**
**[[code](https://github.com/kathrinse/be_great)]**


**[REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers](https://arxiv.org/abs/2302.02041)**

**[Generative Table Pre-training Empowers Models for Tabular Prediction](https://arxiv.org/abs/2305.09696)**
**[[code](https://github.com/ZhangTP1996/TapTap)]**

**[TabuLa: Harnessing Language Models for Tabular Data Synthesis](https://arxiv.org/abs/2310.12746)**
**[[code](https://github.com/zhao-zilong/Tabula)]**

**[Curated LLM: Synergy of LLMs and Data Curation for tabular augmentation in ultra low-data regimes](https://arxiv.org/abs/2312.12112)**

**[TabMT: Generating tabular data with masked transformers](https://arxiv.org/abs/2312.06089)**

**[Elephants Never Forget: Testing Language Models for Memorization of Tabular Data](https://arxiv.org/abs/2403.06644)**
<a id="understanding"></a>

**[Graph-to-Text Generation with Dynamic Structure Pruning](https://arxiv.org/abs/2209.07258)**

**[Plan-then-Seam: Towards Efficient Table-to-Text Generation](https://arxiv.org/abs/2302.05138)**

**[Differentially Private Tabular Data Synthesis using Large Language Models](https://arxiv.org/abs/2406.01457)**

**[Pythia: Unsupervised Generation of Ambiguous Textual Claims from Relational Data](https://iris.unibas.it/bitstream/11563/157086/1/42.SIGMOD2022.pdf)**

### Table understanding
---
#### Numeric Question Answering
**[DocMath-Eval: Evaluating Numerical Reasoning Capabilities of LLMs in Understanding Long Documents with Tabular Data](https://arxiv.org/abs/2311.09805)**

**[Exploring the Numerical Reasoning Capabilities of Language Models: A Comprehensive Analysis on Tabular Data](https://arxiv.org/abs/2311.02216)**

**[TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT](https://arxiv.org/abs/2307.08674)**

#### Question Answering
**[Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning](https://arxiv.org/abs/2301.13808)**
**[[code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/dater)]**

**[PACIFIC: Towards Proactive Conversational Question Answering over Tabular and Textual Data in Finance](https://arxiv.org/abs/2210.08817)**
**[[code](https://github.com/dengyang17/PACIFIC)]**

**[Large Language Models are few(1)-shot Table Reasoners](https://aclanthology.org/2023.findings-eacl.83/)**
**[[code](https://github.com/wenhuchen/tablecot)]**

**[cTBLS: Augmenting Large Language Models with Conversational Tables](https://arxiv.org/abs/2303.12024)**
**[[code](https://github.com/avalab-gt/ctbls)]**

**[Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study](https://arxiv.org/abs/2305.13062)**

**[Large Language Models are Complex Table Parsers](https://aclanthology.org/2023.emnlp-main.914/)**

**[Rethinking Tabular Data Understanding with Large Language Models](https://arxiv.org/abs/2312.16702)**
**[[code](https://github.com/Leolty/tablellm)]**

**[TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT](https://arxiv.org/abs/2307.08674)**

**[Testing the Limits of Unified Sequence to Sequence LLM Pretraining on Diverse Table Data Tasks](https://arxiv.org/abs/2310.00789)**

**[Unified Language Representation for Question Answering over Text, Tables, and Images](https://aclanthology.org/2023.findings-acl.292/)**

**[SUQL: Conversational Search over Structured and Unstructured Data with Large Language Models](https://aclanthology.org/2024.findings-naacl.283/)**
**[[code](https://github.com/stanford-oval/suql)]**

**[TableLlama: Towards Open Large Generalist Models for Tables](https://arxiv.org/abs/2311.09206)**
**[[code](https://github.com/OSU-NLP-Group/TableLlama)]**

**[DIVKNOWQA: Assessing the Reasoning Ability of LLMs via Open-Domain Question Answering over Knowledge Base and Text](https://arxiv.org/abs/2310.20170)**

**[StructGPT: A General Framework for Large Language Model to Reason over Structured Data](https://aclanthology.org/2023.emnlp-main.574/)**
**[[code](https://github.com/RUCAIBox/StructGPT)]**

**[JarviX: A LLM No code Platform for Tabular Data Analysis and Optimization](https://arxiv.org/abs/2312.02213)**

**[CABINET: Content Relevance-based Noise Reduction for Table Question Answering](https://arxiv.org/abs/2402.01155)**
**[[code](https://github.com/Sohanpatnaik106/CABINET_QA)]

**[Traffic Performance GPT (TP-GPT): Real-Time Data Informed Intelligent ChatBot for Transportation Surveillance and Management](https://arxiv.org/abs/2405.03076)**

**[Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow](https://arxiv.org/abs/2306.07209)**
**[[code](https://github.com/zwq2018/Data-Copilot)]**

**[Querying Large Language Models with SQL](https://arxiv.org/pdf/2304.00472)**


#### Text2SQL
**[Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation](https://arxiv.org/abs/2308.15363)**

**[DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction](https://arxiv.org/abs/2304.11015)**
**[[code](https://github.com/mohammadrezapourreza/few-shot-nl2sql-with-prompting)]**

**[C3: Zero-shot Text-to-SQL with ChatGPT](https://arxiv.org/abs/2307.07306)**
**[[code](https://github.com/bigbigwatermalon/C3SQL)]**

**[DBCopilot: Scaling Natural Language Querying to Massive Databases](https://arxiv.org/abs/2312.03463)**
**[[code](https://github.com/tshu-w/DBCopilot?tab=readme-ov-file)]**

**[Bridging the Gap: Deciphering Tabular Data Using Large Language Model](https://arxiv.org/abs/2308.11891)**

**[TableQuery: Querying tabular data with natural language](https://arxiv.org/abs/2202.00454)**
**[[code](https://github.com/abhijithneilabraham/tableQA)]**

**[S2SQL: Injecting Syntax to Question-Schema Interaction Graph Encoder for Text-to-SQL Parsers](https://arxiv.org/abs/2203.06958)**

**[Dynamic hybrid relation network for cross-domain context-dependent semantic parsing](https://arxiv.org/abs/2101.01686)**

**[STAR: SQL Guided Pre-Training for Context-dependent Text-to-SQL Parsing](https://arxiv.org/abs/2210.11888)**

**[SUN: Exploring Intrinsic Uncertainties in Text-to-SQL Parsers](https://arxiv.org/abs/2209.06442)**

**[Towards Generalizable and Robust Text-to-SQL Parsing](https://arxiv.org/abs/2210.12674)**

**[Before Generation, Align it! A Novel and Effective Strategy for Mitigating Hallucinations in Text-to-SQL Generation](https://arxiv.org/abs/2405.15307)**
**[[code](https://github.com/quge2023/TA-SQL)]**


#### Table2Text
**[Robust (Controlled) Table-to-Text Generation with Structure-Aware Equivariance Learning](https://arxiv.org/abs/2205.03972)**
**[[code](https://github.com/luka-group/Lattice)]**


#### Fact Verification
**[Table-based Fact Verification with Salience-aware Learning](https://arxiv.org/abs/2109.04053)**
**[[code](https://github.com/luka-group/Salience-aware-Learning)]**

#### Table Profiling

**[Cocoon: Semantic Table Profiling Using Large Language Models](https://dl.acm.org/doi/abs/10.1145/3665939.3665957)**
**[[code](https://cocoon-data-transformation.github.io/page/profile)]**


#### Table Transformation

**[Relationalizing Tables with Large Language Models: The Promise and Challenges](https://ieeexplore.ieee.org/abstract/document/10555085/)**

#### Entity Matching

**[Disambiguate Entity Matching using Large Language Models through Relation Discovery](https://dl.acm.org/doi/abs/10.1145/3665601.3669844)**
**[[code](https://cocoon-data-transformation.github.io/page/standardize)]**

<a id="dataset"></a>
## Datasets
Please refer to our paper to see relevant methods that benchmark on these datasets. 
<a id="dprediction"></a>
### Prediction Tasks
|Dataset|Dataset Number|Dataset Repo|
| ----- | ------------ | --------------------------- |
OpenML           | 11                          | https://github.com/UW-Madison-Lee-Lab/LanguageInterfacedFineTuning/tree/master/regression/realdata/data | 
Kaggle API       | 169                         | https://github.com/Kaggle/kaggle-api | 
Combo            | 9                           | https://github.com/clinicalml/TabLLM/tree/main/datasets| 
UCI ML           | 20                          | https://github.com/dylan-slack/Tablet/tree/main/data/benchmark/performance| 
DDX              | 10                          | https://github.com/dylan-slack/Tablet/tree/main/data/ddx_data_no_instructions/benchmark |
<a id="dqa"></a>
### Table Understanding Tasks
|Dataset|# Tables|Task Type|Input|Output|Data Source| Dataset Repo                                                                                                  |
| ----- | ------ | ------- | --- | --- | --------- |---------------------------------------------------------------------------------------------------------------|
FetaQA | 10330 | QA | Table Question | Answer | Wikipedia | https://github.com/Yale-LILY/FeTaQA                                                                           | 
WikiTableQuestion | 2108 | QA | Table Question | Answer | Wikipedia | https://ppasupat.github.io/WikiTableQuestions/                                                                | 
NQ-TABLES | 169898 | QA | Question, Table | Answer | Synthetic | https://github.com/google-research-datasets/natural-questions                                                 | 
HybriDialogue | 13000 | QA | Conversation, Table, Reference | Answer | Wikipedia | https://github.com/entitize/HybridDialogue                                                                    | 
TAT-QA  | 2757 | QA | Question, Table | Answer | Financial report | https://github.com/NExTplusplus/TAT-QA                                                                        | 
HiTAB | 3597 | QA/NLG | Question, Table | Answer | Statistical Report and Wikipedia | https://github.com/microsoft/HiTab                                                                            | 
ToTTo | 120000 | NLG | Table | Sentence | Wikipedia  | https://github.com/google-research-datasets/ToTTo                                                             | 
FEVEROUS  | 28800 | Classification | Claim, Table | Label | Common Crawl | https://fever.ai/dataset/feverous.html                                                                        | 
Dresden Web Tables| 125M | Classification | Table | Label | Common Crawl | https://ppasupat.github.io/WikiTableQuestions/                                                                | 
InfoTabs  | 2540 | NLI | Table , Hypothesis | Label | Wikipedia | https://infotabs.github.io/                                                                                   | 
TabFact | 16573 | NLI | Table, Statement | Label | Wikipedia | https://tabfact.github.io/                                                                                    | 
TAPEX  | 1500 | Text2SQL | SQL, Table | Answer | Synthetic | https://github.com/google-research/tapas                                                                      | 
Spider  | 1020 | Text2SQL | Table, Question | SQL | Human annotation | https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download&authuser=0 | 
WIKISQL | 24241 | Text2SQL | Table, Question | SQL, Answer | Human Annotated | https://github.com/salesforce/WikiSQL                                                                         |
BIRD | 12751 | Text2SQL | Table, Question | SQL | Human Annotated | https://bird-bench.github.io/                                                                                 |
Tapilot-Crossing | 5 | Text2Code, QA, RAG | Table, Dialog History, Question, Private Lib, Chart | Python, Private Lib Code, Answer | Human-Agent Interaction | https://tapilot-crossing.github.io/ |

## Survey
**[A Survey on Text-to-SQL Parsing: Concepts, Methods, and Future Directions](https://arxiv.org/abs/2208.13629)**


# Contributing
If you would like to contribute to this list or writeup, feel free to submit a pull request!

