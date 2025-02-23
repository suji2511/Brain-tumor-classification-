# Brain Tumor Classification

This project implements a machine learning model to classify brain tumor MRI images into four categories: glioma tumor, meningioma tumor, pituitary tumor, and no tumor. The implementation uses a Random Forest Classifier and achieves competitive accuracy on the test set.

## Dataset

The project uses the Brain Tumor MRI Dataset from Kaggle, which contains MRI scans categorized into four classes:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

## Requirements

```
numpy
opencv-python
scikit-learn
kagglehub
matplotlib
seaborn
pillow
```

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/suji2511/Brain-tumor-classification
cd Brain-tumor-classification
```

2. Run the classifier:
```bash
python brain_tumor_classifier.py
```

## Features

- Automatic dataset download using kagglehub
- Image preprocessing and normalization
- Random Forest Classification
- Performance metrics visualization
- Sample prediction visualization

## Model Performance

The model generates:
- Accuracy metrics
- Confusion matrix
- Classification report
- Sample predictions visualization

## Project Structure

```
Brain-tumor-classification/
│
├── brain_tumor_classifier.py     # Main classifier implementation
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
├── confusion_matrix.png         # Generated confusion matrix
└── sample_predictions.png       # Generated sample predictions
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
