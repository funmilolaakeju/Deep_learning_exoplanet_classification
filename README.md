# Deep Learning Exoplanet Classification

This project implements deep learning models for astrophysics classification using NASA Kepler Object of Interest data from the NASA Exoplanet Archive. The project builds and compares Multi Layer Perceptron and Convolutional Neural Network models for multiclass exoplanet disposition classification.

The pipeline includes scientific data download, preprocessing, feature scaling, model training, evaluation, visualization, and model persistence.

## Live Demo

This project includes a deployed Streamlit web application for interactive exoplanet classification.

Users can enter scientific Kepler feature values and receive predicted probabilities for:
- CANDIDATE
- CONFIRMED
- FALSE POSITIVE


# Features

- Real astrophysics dataset from NASA Exoplanet Archive
- Kepler Object of Interest classification
- Multi Layer Perceptron model
- 1D Convolutional Neural Network model
- Scientific feature preprocessing
- Standard scaling
- Class imbalance handling using class weights
- Accuracy, precision, recall, and F1 score evaluation
- Confusion matrix visualization
- Training history visualization
- Model saving and reproducibility
- Jupyter and Google Colab compatible

---
# Data Source

This project uses publicly available Kepler Object of Interest data from the NASA Exoplanet Archive operated by NASA/IPAC.
Official source:https://exoplanetarchive.ipac.caltech.edu/
The dataset is accessed programmatically through the NASA Exoplanet Archive TAP service for educational and research purposes.


Target variable:

```text
koi_disposition
```

Classes:

- CONFIRMED
- CANDIDATE
- FALSE POSITIVE

Scientific features used:

- koi_period
- koi_impact
- koi_duration
- koi_depth
- koi_prad
- koi_teq
- koi_insol
- koi_model_snr
- koi_steff
- koi_slogg
- koi_srad

---

# Repository Structure

```text
deep-learning-exoplanet-classification/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/deep-learning-exoplanet-classification.git

cd deep-learning-exoplanet-classification
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment.

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

```text
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
tensorflow>=2.16.0
requests>=2.31.0
```

---

# Run the Project

Train the MLP model:

```bash
python main.py --model mlp
```

Train the CNN model:

```bash
python main.py --model cnn
```

Train with custom epochs:

```bash
python main.py --model mlp --epochs 100
```

For Google Colab or Jupyter notebooks, replace the final block with:

```python
args = parse_args([])
run_pipeline(args)
```

---

# Experiment Configuration

| Parameter | Value |
|---|---|
| Dataset | NASA Exoplanet Archive KOI Table |
| Models | MLP, CNN |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 100 |
| Batch Size | 32 |
| Scaling Method | StandardScaler |
| Train Test Split | 80/20 |
| Random Seed | 42 |
| Class Imbalance Handling | Balanced Class Weights |

---

# Results

The improved MLP model achieved strong multiclass classification performance on real astrophysics data.

Evaluation Metrics:

```text
Accuracy: 76.05%
Precision: 81.13%
Recall: 76.05%
F1 Score: 77.11%
```

The class-weighted training strategy significantly improved recall for the CANDIDATE class, increasing balanced classification performance across astrophysical categories.

Classification Summary:

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| CANDIDATE | 0.50 | 0.77 | 0.60 |
| CONFIRMED | 0.81 | 0.86 | 0.84 |
| FALSE POSITIVE | 0.94 | 0.70 | 0.80 |

---

# Output Files

Generated outputs are automatically saved:

```text
results/models/exoplanet_mlp_model.keras

results/plots/training_history.png

results/plots/confusion_matrix.png

data/koi_exoplanet_data.csv
```

---

# Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- NASA Exoplanet Archive API

---

# Resume Bullet

Built CNN and MLP models for astrophysics classification using NASA Kepler Object of Interest data, improving CANDIDATE class recall from 30% to 77% using class-weighted deep learning.

---

# Future Improvements

Potential future extensions include:

- Transformer based architectures
- Hyperparameter optimization
- Explainable AI feature attribution
- Streamlit dashboard deployment
- Docker containerization
- FastAPI inference API
- Ensemble learning methods

---

# License

This project is intended for educational and research purposes.