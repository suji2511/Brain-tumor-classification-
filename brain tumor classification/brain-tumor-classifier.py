# brain_tumor_classifier.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import kagglehub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

class BrainTumorClassifier:
    def __init__(self, data_path=None):
        """
        Initialize the Brain Tumor Classifier
        Args:
            data_path (str): Path to the dataset. If None, downloads from Kaggle
        """
        self.data_path = data_path
        self.tumor_categories = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]
        self.model = None
        
    def download_dataset(self):
        """Download the dataset from Kaggle"""
        return kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    def load_and_process_data(self):
        """Load and process the MRI images"""
        if not self.data_path:
            self.data_path = self.download_dataset()
            self.data_path = "/root/.cache/kagglehub/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1"

        data = []
        labels = []
        
        for main_category in ["Testing", "Training"]:
            main_path = os.path.join(self.data_path, main_category)
            subcategories = os.listdir(main_path)
            
            for subcategory in subcategories:
                subcategory_path = os.path.join(main_path, subcategory)
                label = subcategories.index(subcategory)
                
                for img_name in os.listdir(subcategory_path):
                    try:
                        img_path = os.path.join(subcategory_path, img_name)
                        img = Image.open(img_path).convert("L")
                        img = img.resize((128, 128))
                        img_array = np.array(img)
                        data.append(img_array)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {img_name}: {e}")
        
        self.data = np.array(data) / 255.0
        self.labels = np.array(labels)
        return self.data, self.labels

    def prepare_data(self):
        """Prepare and split the data for training"""
        data_flattened = self.data.reshape(self.data.shape[0], -1)
        return train_test_split(
            data_flattened, self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels
        )

    def train_model(self, X_train, y_train):
        """Train the Random Forest classifier"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and return metrics"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.tumor_categories)
        return accuracy, conf_matrix, report, y_pred

    def plot_confusion_matrix(self, conf_matrix):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                   xticklabels=self.tumor_categories,
                   yticklabels=self.tumor_categories)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig('confusion_matrix.png')
        plt.close()

    def display_sample_predictions(self, X_test, y_test, y_pred, num_samples=5):
        """Display and save sample predictions"""
        plt.figure(figsize=(15, 10))
        for i in range(num_samples):
            index = np.random.randint(0, len(X_test))
            img = X_test[index].reshape(128, 128)
            true_label = self.tumor_categories[y_test[index]]
            predicted_label = self.tumor_categories[y_pred[index]]
            
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
            plt.axis("off")
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png')
        plt.close()

def main():
    # Initialize classifier
    classifier = BrainTumorClassifier()
    
    # Load and process data
    data, labels = classifier.load_and_process_data()
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data()
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    # Train model
    classifier.train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, conf_matrix, report, y_pred = classifier.evaluate_model(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    # Generate visualizations
    classifier.plot_confusion_matrix(conf_matrix)
    classifier.display_sample_predictions(X_test, y_test, y_pred)

if __name__ == "__main__":
    main()
