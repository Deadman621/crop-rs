# Crop Recommender System (Crop-RS)

A multi-modal recommendation system that combines computer vision and environmental data to provide precise, geologically compatible crop suggestions.

## Overview
This project implements a dual-input pipeline:
1. **Image Model:** A ResNet50 CNN that identifies soil type from a photo.
2. **Tabular Model:** An XGBoost classifier that predicts suitable crops based on environmental parameters (N, P, K, pH, Temperature, Humidity, Rainfall).

The system cross-references these predictions with a **Knowledge Domain** to ensure that recommended crops are biologically compatible with the detected soil type.

## Project Structure
```
.
├── artifacts/              # Model checkpoints (best_resnet_soil.pth)
├── data/
│   ├── soil_images/        # Original and Augmented (CyAUG) soil datasets (gitignored due to large file size)
│   └── tabular/            # Crop recommendation CSV data
├── rs.ipynb                # Main development notebook
├── requirements.txt        # Project dependencies
└── run-crop-torch.sh       # Execution script
```

## Technical Details

### 1. Image Model (Soil Classification)
- **Architecture:** ResNet50 (Transfer Learning from ImageNet).
- **Dataset:** 6,286 images across 7 soil types (Alluvial, Arid, Black, Laterite, Mountain, Red, Yellow).
- **Augmentation:** Hybrid strategy using pre-augmented data (CyAUG) and real-time transforms (Flips, Color Jitter).
- **Performance:** 86% Test Accuracy.

### 2. Tabular Model (Crop Recommendation)
- **Algorithm:** XGBoost Classifier.
- **Features:** N, P, K, Temperature, Humidity, pH, Rainfall.
- **Objective:** `multi:softprob` (provides ranked probabilities for 22 unique crops).
- **Evaluation Metrics:** `mlogloss` (Multi-class Log Loss) and `merror` (Classification Error).
- **Optimization:** Histogram-based tree method (`tree_method="hist"`) for efficiency.
- **Performance:** ~98% Validation Accuracy.

### 3. Integration Pipeline
- **Precondition Check:** Validates environmental inputs against realistic agricultural limits.
- **Multi-Modal Inference:** Parallel execution of soil identification and crop prediction.
- **Knowledge-Based Filtering:** Final verification of crop-soil compatibility to eliminate mismatches.

## Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Notebook:**
   Open `rs.ipynb` in VS Code or Jupyter Lab to see the full EDA, training, and evaluation process.

## Dataset Links:

- [Tabular Dataset](https://www.kaggle.com/code/abdullahbasit/soil-classification?select=Crop_recommendation.csv)
- [Soil Images](https://www.kaggle.com/code/abdullahbasit/soil-classification/input)

## Impact
This system empowers precision agriculture by reducing crop failure risks through multi-modal verification, ensuring that recommendations are both climatically and geologically sound.

