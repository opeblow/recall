<div align="center">

# RESEARCH

*A continual learning research framework for studying and comparing different continual learning methods.*

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stars](https://img.shields.io/github/stars/research-cl/research)
![Last Commit](https://img.shields.io/github/last-commit/research-cl/research/main)
![Research Topic](https://img.shields.io/badge/research_topic-Continual%20Learning-purple)

</div>

```text
    .--.        .--.        .--.        .--.        .--.      
   (    )      (    )      (    )      (    )      (    )     
  (  )  )    (  )  )    (  )  )    (  )  )    (  )  )    
 (  )(  )    (  )(  )    (  )(  )    (  )(  )    (  )(  )   
  '.  .'      '.  .'      '.  .'      '.  .'      '.  .'    
   '._.'      '._.'      '._.'      '._.'      '._.'      

     NEURAL NETWORK CONTINUAL LEARNING FRAMEWORK
     
  [Input] -> [Dense] -> [ReLU] -> [Dense] -> [Output]
       \        |          |         |        /
        \       v          v         v       /
         `---> [Memory Buffer] <-------'
                (Prevent Forgetting)
```

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Continual Learning Methods](#continual-learning-methods)
- [Metrics](#metrics)
- [Configuration](#configuration)
- [Built With](#built-with)
- [License](#license)
- [Footer](#footer)

---

## Overview

**RESEARCH** is a comprehensive continual learning research framework designed for studying and comparing different continual learning methods. This framework provides implementations of state-of-the-art techniques to mitigate catastrophic forgetting in neural networks when learning new tasks sequentially.

Continual learning (also known as lifelong learning) is a challenging paradigm where models must learn from a stream of tasks without forgetting previously acquired knowledge. This framework implements multiple approaches to address this problem.

---

## Features

- **Multiple CL Methods**: Implementations of Naive, EWC, Replay, GEM, and Hybrid approaches
- **Benchmark Runner**: Standardized evaluation across all methods
- **Metrics Tracking**: Comprehensive metrics including accuracy, forgetting, and transfer
- **Experiment Logging**: Integrated Weights & Biases support for experiment tracking
- **Interactive Dashboard**: Streamlit-based visualization tool for results
- **Configurable**: YAML-based configuration for all hyperparameters
- **Extensible**: Easy to add new continual learning methods

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Weights & Biases account (for logging)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/continuum.git
   cd research
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Add your Weights & Biases API key to the `.env` file:
   ```
   WAND_API_KEY=your_api_key_here
   ```

---

## Usage

### Running Benchmarks

Run a benchmark with a specific method:

```bash
python run_benchmark.py --method naive --num_tasks 5 --epochs 10
```

**Command-line options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--method` | Method to use (naive/ewc/replay/gem/hybrid) | naive |
| `--num_tasks` | Number of tasks | 5 |
| `--epochs` | Epochs per task | 10 |
| `--lr` | Learning rate | 0.001 |
| `--config` | Path to custom config JSON | None |
| `--yaml_config` | Path to custom config YAML | None |

### Available Methods

- **naive**: Naive baseline (no continual learning)
- **ewc**: Elastic Weight Consolidation
- **replay**: Experience Replay
- **gem**: Gradient Episodic Memory
- **hybrid**: Hybrid method (EWC + Replay)

### Using the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run research/dashboard/app.py
```

Upload CSV results to visualize performance metrics.

---

## Continual Learning Methods

### Naive (Baseline)
Simple baseline that trains on each task sequentially without any regularization. This method suffers from catastrophic forgetting as it tends to overwrite weights important for previous tasks.

### Elastic Weight Consolidation (EWC)
EWC adds a regularization term to the loss that penalizes changes to important parameters, as measured by the Fisher information matrix. This helps preserve knowledge from previous tasks.

### Experience Replay
This method stores a small subset of examples from previous tasks and replays them during training on new tasks. This helps prevent forgetting by maintaining a memory buffer of past experiences.

### Gradient Episodic Memory (GEM)
GEM stores a small buffer of examples from previous tasks and constrains gradients to prevent interference with previous tasks. It projects gradients to ensure they don't increase loss on past tasks.

### ResearchHybrid
A novel hybrid method that combines:
- EWC regularization on backbone weights
- Selective replay using highest softmax entropy samples (hardest examples)
- Dynamic task heads (separate output head per task)

---

## Metrics

The benchmark computes the following metrics:

| Metric | Description |
|--------|-------------|
| **Average Accuracy** | Mean accuracy across all tasks |
| **Backward Transfer** | How much learning new tasks hurts previous tasks |
| **Forward Transfer** | How much learning new tasks helps future tasks |
| **Forgetting** | Maximum drop in performance on each task |

---

## Configuration

All hyperparameters are configurable via `config.yaml`. You can also override values via command-line arguments or a custom config file.

### Default Configuration

```yaml
# Model Configuration
model:
  input_dim: 784
  hidden_dim: 256
  output_dim: 10

# Training Configuration
training:
  epochs: 10
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001

# EWC Method Configuration
ewc:
  ewc_lambda: 5000

# Replay Method Configuration
replay:
  memory_size: 100

# GEM Method Configuration
gem:
  memory_size: 200

# ResearchHybrid Method Configuration
research_hybrid:
  ewc_lambda: 1000
  replay_size: 300
  heads_lr: 0.01
```

---

## Project Structure

```
research/
├── config.yaml                 # Configuration file
├── run_benchmark.py            # Benchmark runner script
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # Project documentation
├── research/
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── base.py            # Base class for CL methods
│   │   ├── naive.py           # Naive baseline
│   │   ├── ewc.py             # Elastic Weight Consolidation
│   │   ├── replay.py          # Experience Replay
│   │   ├── gem.py             # Gradient Episodic Memory
│   │   └── research_hybrid.py # Hybrid method
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── runner.py          # Benchmark runner
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py             # Streamlit dashboard
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration loader
│       └── logger.py          # Logging utilities
```

---

## Built With

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Weights & Biases](https://img.shields.io/badge/Weights%20&%20Biases-FFBE1A?style=for-the-badge&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

</div>

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">

---

Made with purpose by **MOBOLAJI OPEYEMI BOLATITO**

---

</div>
