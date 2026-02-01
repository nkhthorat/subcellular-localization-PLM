PLM-based Protein Subcellular Localization (Diatoms / Complex Plastids) 
======================================================================

This repo implements a sequence-only pipeline for protein subcellular localization using
frozen protein language model (PLM) embeddings + a small MLP classifier, evaluated with
homology-aware cross-validation.

WHAT THE PIPELINE DOES
----------------------
1) Load & clean data from a spreadsheet/CSV containing sequences and labels
2) Homology-aware CV splits via GraphPart (optional but enabled by default)
3) Embed N-terminus with a PLM backbone (ProtT5 or ESM-1b), optionally averaging the last k layers
4) Pool residue embeddings into a fixed-size vector (multiple biologically motivated pooling options, mean_prefix_50 by default)
5) Train an MLP on pooled features and compute out-of-fold (OOF) metrics and reports

REPOSITORY LAYOUT (KEY FILES)
-----------------------------
- run_experiment.py : main entrypoint (CLI) to run an experiment end-to-end

Core modules:
- config.py     : experiment configuration dataclass
- data.py       : data loading + sequence cleaning
- graphpart.py  : GraphPart wrapper + CV split building
- embeddings.py : PLM inference + pooling to fixed vectors
- pooling.py    : pooling strategies over residue embeddings
- model.py      : MLP classifier
- train.py      : training loop + early stopping
- eval.py       : OOF metrics + reports
- utils.py      : logging, seeding, helpers

TESTED ENVIRONMENT (REFERENCE)
------------------------------
Developed/tested on Google Colab with a Tesla T4 GPU.

Colab shortcut (T4 runtime)
--------------------------
If you run this project on Google Colab with a T4 GPU, you can usually skip installing
the full Python requirements because Colab already ships with most dependencies
(e.g., torch/transformers/numpy/pandas/sklearn). In that case, you only need to:

1) Upload/clone the repo WITH the same folder structure and the data file.
2) Install the system + CLI dependencies below:

# System package for EMBOSS (provides `needleall`)
!apt-get update -y
!apt-get install -y emboss

# Python package for GraphPart (provides `graphpart` CLI)
!pip install graph-part==1.0.2

(Quick check)
!which needleall
!graphpart --help

Local install / other environments
---------------------------------
For local machines or clean virtual environments, install Python dependencies using:

- pip install -r requirements.txt
- plus ONE of:
  - pip install -r requirements-torch-cpu.txt
  - pip install -r requirements-torch-cu126.txt  (CUDA 12.6)

System dependency (GraphPart needle mode):
- EMBOSS must be installed system-wide so `needleall` is on PATH.

Environment details
-------------------------------
- Python: 3.12.12
- PyTorch: 2.9.0+cu126
- CUDA available: True
- CUDA version: 12.6
- GPU: Tesla T4

REQUIREMENTS
------------
Python packages:
- See requirements.txt (core)
- Install ONE of the torch requirement files depending on your machine

System dependency (NOT pip):
- EMBOSS (provides `needleall`) is required if you use GraphPart "needle" mode.

Linux (Ubuntu/Debian):
  sudo apt-get update
  sudo apt-get install -y emboss

macOS (Homebrew):
  brew install emboss

DATA FORMAT
-----------
Provide an .xlsx/.xls or .csv file with two columns:
- Protein sequence
- Categories (class label)

At load time, these are renamed internally to: sequence, label
Rows with missing sequences/labels are dropped. A stable seq_id is assigned and a cleaned
sequence (sequence_clean) is produced.

INSTALL
-------
1) Create and activate a virtual environment (recommended):
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip

2) Install core requirements:
   pip install -r requirements.txt

3) Install PyTorch (pick ONE):
   pip install -r requirements-torch-cpu.txt
   # OR, for CUDA 12.6:
   pip install -r requirements-torch-cu126.txt

4) (Optional, for GraphPart needle mode) Install EMBOSS so `needleall` is available.

QUICK VERIFY
------------
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda_ver:', torch.version.cuda)"
needleall -h

RUN AN EXPERIMENT
-----------------
Example:

python run_experiment.py \
  --data data.xlsx \
  --save-dir runs/run1 \
  --backbone prott5 \
  --nterm 250 \
  --pooling mean_prefix_50 \
  --seed 100

OUTPUTS
-------
The --save-dir folder will contain:
- label_encoder.pkl
- graphpart/                      (if enabled)
- oof_confusion_matrix.csv
- oof_report.txt
- summary.json

NOTES
-----
- GPU is strongly recommended for PLM embedding speed; CUDA will be used automatically if available.
- If you want to disable GraphPart/homology-aware splits, you will need to modify run_experiment.py
  (it currently always calls GraphPart split generation).
