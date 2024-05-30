#  Doubly Disentangled Intrusion Detection
=======
# DIDS-MFL

A disentangled intrusion detection method to handle various intrusion detection scenarios, e.g. known attacks, unknown attacks, and few-shot attacks.


This repository contains the implementation of a sophisticated intrusion detection system (IDS) tailored for the Internet of Things (IoT) environment, which we called DIDS-MFL. DIDS-MFS has two critical components: DIDS and MFS. DIDS is a novel method that aims to tackle the performance inconsistent issues through two-step feature disentanglements and a dynamic graph diffusion scheme, and MFS is a plug-and-play module few-shot intrusion detection module by multi-scale information fusion. Equipped with DIDS-MFL, the administrators can effectively identify various attacks in encrypted traffic, including known, unknown ones, and the few-shot threats that are not easily detected. **To the best of our knowledge, DIDS-MFL takes the first step to achieve intrusion detection in various scenarios.**

![43a443e0befbe34d6aed89cd354b439](https://github.com/qcydm/DIDS-MFL/assets/42266769/57c3b662-a3c1-485e-8d21-028b55e71529)

## Key Features

- **Dynamic Graph Diffusion**: Employs a dynamic graph diffusion scheme for spatial-temporal aggregation in evolving data streams.
- **Feature Disentanglement**: Utilizes a two-step feature disentanglement process to address the issue of entangled distributions of flow features.
- **Few-Shot Learning Module**: Integrates a few-shot learning approach for effective detection of novel intrusion types with limited instance data.
- **Memory Model**: Incorporates a memory model to generate representations that highlight attack-specific features.
- **Real-Time Detection**: Capable of performing real-time anomaly detection with high accuracy and speed.

## Getting Started

### Prerequisites

-numpy>=1.18.5
-pandas>=1.0.5
-scikit-learn>=0.23.1
-tqdm>=4.46.1
-torch>=1.6.0
-torch-geometric>=2.4.0
-scipy>=1.4.1
-z3-solver==4.12.2.0

### Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```
For the torch-geometric library, multiple files need to be downloaded. You can visit the following website to select the appropriate version that matches your environment: (https://data.pyg.org/whl/). After downloading, you can install the specific version with the following command:

```bash
pip install torch_geometric==2.0.4 -i https://pypi.doubanio.com/simple
```

You can choose the version that best suits your needs.

### Usage

To train and evaluate the model, run the following command:

```bash
python main.py
```

This will start the training process, including the pre-training, meta-training, and meta-testing phases.


### Base Models

| Model Name            | Role/Function                                                                                 |
|-----------------------|----------------------------------------------------------------------------------------------|
| TGNMemory             | Manages the temporal dynamics and memory state of nodes in the graph.                         |
| MGD (Multi-Graph Diffusion) | Allows information propagation across different graph structures.                          |
| MLPPredictor          | A multilayer perceptron for binary and multiple class predictions.                             |
| SelfExpr              | Enhances learning capabilities through self-expression mechanisms.                             |
| Loss Function         | Custom function designed to optimize model performance for the specific task.                 |



### Training Details

| Aspect                | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| Data Preparation      | The dataset is preprocessed and filtered to ensure the inclusion of the required number of classes. |
| Model Architecture    | Utilizes `TGNMemory` for node memory state and `MGD` for updating node representations.         |
| Model Parameters      | Epsilon:0.3, Beta:1, gamma:0.1, Eta=0.1, alpha=0.1    |
| Optimizer             | Adam optimizer is used with a learning rate of 0.0001.                                         |
| Loss Function         | Custom loss function combining binary and multiple output losses.                              |
| Evaluation Metrics    | F1 Score, Normalized Mutual Information (NMI), Precision, and Recall are used for evaluation.   |
| Training Procedure    | Training is monitored with a tqdm progress bar and iterated until convergence.                 |


### Datasets

Benchmark Datasets
The system is trained and tested on several benchmark datasets, which are commonly used in the field of network traffic analysis and intrusion detection. These datasets provide a rich source of labeled data for training and evaluating models.

You can change datasets in main.py:
```bash
    data_all = torch.load(datasets)
```

- CIC-ToN-IoT: A dataset designed for evaluating network-based intrusion detection systems in the context of IoT networks.
- CIC-BoT-IoT: Another IoT-focused dataset that simulates botnet attacks in a network environment.
- EdgeIIoT: A dataset collected from real-world IoT devices and networks, focusing on edge computing scenarios.
- NF-UNSW-NB15-v2: A network flow dataset containing modern normal and attack traffic traces from a university network.
- NF-CSE-CIC-IDS2018-v2: A dataset that combines network flows and system logs to provide a comprehensive view of network traffic.
- NF-UQ-NIDS：A dataset from the University of Queensland for testing NIDS, featuring various network traffic scenarios.
- NF-BoT-IoT-v2：An updated dataset focusing on IoT botnet attacks, providing refined data for network security research.


### Using Your Own Data
You can also use your own data with our system by datasets.py. Please make sure that you keep the following column information：
column information = TemporalData(  
    src=src,  
    dst=dst,  
    src_layer=src_layer,  
    dst_layer=dst_layer,  
    t=t,  
    dt=dt,  
    msg=msg,  
    label=label,  
    attack=attack  
)  
  

### Performance Metrics

The effectiveness of the system is evaluated using the following metrics:

- F1 Score
- Normalized Mutual Information
- Precision and Recall


## Baselines 

we select 11 few-shot learning models to incorporate into 3D-IDS as baselines, including 3 meta-learning based models (i.e., MBase, MTL, TEG), where TEG is designed for graph-structure-based few-shotlearning, 4 augmentation-based models (i.e., CLSA,
ESPT, ICI, KSCL], where CLSA and ESPT are based on contrastive augmentation, ICI and KSCL are based on instance augmentation), 4 metric learning-based models (i.e., BSNet, CMFSL, TAD, PCWPK).


### Baseline Models for 3D-IDS

This project incorporates a series of baseline models for few-shot learning in the context of 3D Intrusion Detection Systems (3D-IDS). Below is a brief description of each model type:

#### Meta-Learning Models
- **MBase**: A fundamental meta-learning model that serves as a basic benchmark for comparison.
- **MTL**: A Multi-Task Learning approach that enhances learning efficiency by leveraging shared features across multiple tasks.
- **TEG**: Tailored for graph-structured data, this model utilizes graph embedding techniques for few-shot learning scenarios.

#### Augmentation-Based Models
- **CLSA**: Employs contrastive augmentation to enhance the model's ability to distinguish between samples.
- **ESPT**: Builds upon contrastive augmentation with different strategies for further performance optimization.
- **ICI**: Utilizes instance augmentation to increase sample diversity and improve model generalization.
- **KSCL**: Another instance augmentation-based model, with unique strategies for few-shot learning.

#### Metric Learning-Based Models
- **BSNet**: Focuses on learning sample distances to bring similar samples closer and dissimilar ones further apart in feature space.
- **CMFSL**: A specialized metric learning model that include innovative feature extraction or similarity measurement techniques.
- **TAD**: Involves adaptive distance learning to better fit the few-shot learning context.
- **PCWPK**: Based on pairwise or contrastive learning approaches to improve accuracy in few-shot scenarios.

#### How to Test Baseline Models
To test any of the baseline models, you can run the following command in your terminal, replacing the method name accordingly:
```bash
python [Method].py
```
For example, to test the ICI model, you would use:
```bash
python ICI.py
```
>>>>>>> 94462a6 (first commit)
