# FART Evaluation Pipeline

This repository contains a Jupyter notebook for evaluating FART (Flavor Aroma Recognition Task) models. The notebook trains a RoBERTa classifier on SMILES molecular representations and provides comprehensive evaluation metrics including ROC curves, confusion matrices, and accuracy metrics.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Notebook](#running-the-notebook)
- [Output](#output)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.11 or 3.12 (recommended; Python 3.14+ may have compatibility issues with some packages)
- Git (for cloning the repository)
- Access to HuggingFace models (for downloading pre-trained models)

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fart-chemberta-700k-augmented-smiles
```

### 2. Create and Activate Python Virtual Environment

```bash
# Create a virtual environment (using Python 3.11 or 3.12 is recommended)
python3.11 -m venv venv
# or
python3.12 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install all dependencies
pip install -r requirements.txt
```

**Note:** If you encounter build errors, ensure you have the latest versions of pip, setuptools, and wheel installed. Some packages may require compilation, so make sure you have the necessary build tools installed on your system.

### 3. Verify Installation

```bash
python -c "import torch; import transformers; import rdkit; print('All packages installed successfully!')"
```

### 4. Verify Dataset Structure

Ensure your dataset directory structure is correct:

```
fart-chemberta-700k-augmented-smiles/
├── dataset/
│   └── splits/
│       ├── fart_train.csv
│       ├── fart_val.csv
│       └── fart_test.csv
├── fart_evaluate.ipynb
├── requirements.txt
└── venv/          (created after setup)
```

## Configuration

Before running the notebook, configure the parameters in the **⚙️ Configuration** section (the second code cell):

### Key Configuration Parameters

```python
# Model and Tokenizer
model_checkpoint = "seyonec/SMILES_tokenized_PubChem_shard00_160k"  # HuggingFace model ID or local path
tokenizer_checkpoint = None  # If None, uses model_checkpoint

# Data paths (relative to notebook directory)
data_dir = "dataset/splits"  # Directory containing CSV files

# Training parameters
num_train_epochs = 2
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
max_length = 512  # Maximum sequence length for tokenization
run_name = "fart_evaluation"  # Name for this training run

# Augmentation settings
augmentation = False  # Set to True to enable SMILES augmentation
augmentation_numbers = [10, 10, 10, 10, 10]  # Augmentation per class
```

### Configuration Options Explained

- **`model_checkpoint`**: HuggingFace model identifier (e.g., `"seyonec/SMILES_tokenized_PubChem_shard00_160k"`) or path to a local model directory
- **`tokenizer_checkpoint`**: Optional separate tokenizer path. If `None`, uses `model_checkpoint`
- **`data_dir`**: Relative path to the directory containing `fart_train.csv`, `fart_val.csv`, and `fart_test.csv`
- **`num_train_epochs`**: Number of training epochs
- **`per_device_train_batch_size`**: Batch size for training (adjust based on GPU memory)
- **`per_device_eval_batch_size`**: Batch size for evaluation
- **`max_length`**: Maximum sequence length for tokenization (512 is standard for RoBERTa)
- **`augmentation`**: Enable/disable SMILES augmentation (increases dataset size)
- **`augmentation_numbers`**: Number of augmented SMILES per class: `[bitter, sour, sweet, umami, undefined]`

## Running the Notebook

### 1. Start Jupyter Lab

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install jupyter if not already installed
pip install jupyter jupyterlab

# Start Jupyter Lab
jupyter lab
```

### 2. Open the Notebook

Navigate to `fart_evaluate.ipynb` in Jupyter Lab and open it.

### 3. Configure Parameters

Edit the configuration cell (second code cell) with your desired settings.

### 4. Run All Cells

You can either:
- Run cells sequentially by clicking "Run" for each cell
- Run all cells at once: `Kernel` → `Restart Kernel and Run All Cells`

**Note:** Training may take a while depending on:
- Dataset size
- Number of epochs
- Batch size
- Hardware (CPU vs GPU)

### 5. Monitor Progress

The notebook will display:
- Configuration summary
- Data loading progress
- Training progress (loss, accuracy)
- Validation results
- Test set metrics
- Visualizations (confusion matrix, ROC curves)

## Output

The notebook displays all results inline without saving files. You'll see:

### Metrics

1. **Overall Metrics**:
   - Accuracy
   - Macro Precision, Recall, F1 Score
   - Weighted Precision, Recall, F1 Score

2. **Per-Class Metrics**:
   - Precision, Recall, F1 Score, and Support for each class (bitter, sour, sweet, umami, undefined)

3. **Metrics Summary Table**: A formatted table with all per-class metrics

### Visualizations

1. **Confusion Matrix**: Heatmap showing predicted vs actual labels
2. **ROC Curves**: ROC curves for each class with AUC scores
3. **Ensemble Voting Results** (if augmentation is enabled): Additional metrics and ROC curves for ensemble predictions

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` for packages like `rdkit`, `transformers`, etc.

**Solution**:
```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt

# If rdkit installation fails, try installing it separately:
# pip install rdkit

# On macOS, if pip installation fails, you can try using Homebrew:
# brew install rdkit
# pip install rdkit-pypi
```

#### 2. CUDA/GPU Issues

**Problem**: CUDA out of memory or GPU not detected

**Solution**:
- Reduce `per_device_train_batch_size` and `per_device_eval_batch_size`
- Use CPU by setting device in the notebook (add after imports):
  ```python
  import torch
  device = torch.device("cpu")  # Force CPU usage
  ```
- Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

#### 3. Dataset Not Found

**Problem**: `FileNotFoundError` when loading CSV files

**Solution**:
- Verify `data_dir` path is correct relative to the notebook location
- Ensure CSV files exist: `fart_train.csv`, `fart_val.csv`, `fart_test.csv`
- Check file permissions

#### 4. HuggingFace Model Download Issues

**Problem**: Cannot download model from HuggingFace

**Solution**:
- Check internet connection
- Verify model name is correct
- If using a local model, ensure the path is correct
- For private models, you may need to authenticate:
  ```python
  from huggingface_hub import login
  login()
  ```

#### 5. RDKit SMILES Parsing Errors

**Problem**: Errors during SMILES augmentation

**Solution**:
- This is usually due to invalid SMILES strings in the dataset
- The notebook will skip invalid SMILES, but you may want to clean your dataset
- Check the dataset for malformed SMILES strings

#### 6. Memory Issues

**Problem**: Out of memory during training

**Solution**:
- Reduce batch size (`per_device_train_batch_size`)
- Reduce `max_length` (e.g., from 512 to 256)
- Disable augmentation if enabled
- Close other applications to free up memory

#### 7. Package Installation/Build Errors

**Problem**: Errors during `pip install -r requirements.txt` (e.g., compilation errors)

**Solution**:
- Ensure you're using Python 3.11 or 3.12 (Python 3.14+ may have compatibility issues)
- Upgrade pip, setuptools, and wheel: `pip install --upgrade pip setuptools wheel`
- On macOS, ensure Xcode command line tools are installed: `xcode-select --install`
- On Linux, install build dependencies: `sudo apt-get install build-essential python3-dev` (Ubuntu/Debian) or equivalent for your distribution

### Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Verify all dependencies are installed correctly
3. Ensure your dataset format matches the expected structure
4. Check that configuration parameters are valid

## Dataset Format

The CSV files should contain at minimum:
- `Canonicalized SMILES`: SMILES string representation of molecules
- `Canonicalized Taste`: Taste label (one of: bitter, sour, sweet, umami, undefined)

Optional columns:
- `Standardized SMILES`: Used for ensemble voting when augmentation is enabled

## Notes

- The notebook uses a temporary output directory (`./temp_training_output`) for training checkpoints. This directory is created automatically.
- All visualizations are displayed inline in the notebook and are not saved to files.
- Training progress is logged to `./temp_training_output/logs/` (can be viewed with TensorBoard if needed).
- The model checkpoints are saved during training but the best model is automatically loaded for evaluation.

## License

[Add your license information here]

## Citation

If you use this code, please cite the relevant papers:
- FART dataset paper
- Model checkpoint paper (e.g., SMILES tokenized PubChem)
- Any other relevant citations
