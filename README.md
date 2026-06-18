# AI-Based Fabric Defect Classification using Dual-Backbone Convolutional Neural Network
Fabric defect detection is one of the most critical tasks in textile quality control. Manual inspection is time-consuming, expensive, and highly prone to human error, especially in large-scale industrial production where even minor defects such as holes, stains, broken stitches, or thread errors can lead to significant financial losses.
This project presents an AI-powered fabric defect classification system using a **Dual-Backbone Convolutional Neural Network (CNN)** that combines **EfficientNetB3** and **MobileNetV2** to classify fabric defects into 12 categories. The hybrid architecture leverages the deep hierarchical feature extraction capability of EfficientNetB3 and the lightweight texture-sensitive representation of MobileNetV2.
The model was trained on a hybrid dataset created by combining the Kaggle Fabric Defect Dataset and TILDA Fabric v2 dataset. To improve robustness and address class imbalance, data augmentation, class weighting, and Sparse Categorical Focal Loss were used.
The final model achieved **98.83% test accuracy**, demonstrating strong performance for real-world industrial defect detection.


## Dataset Source
This project uses a hybrid dataset built from:
* Kaggle Fabric Defect Dataset(Available: https://www.kaggle.com/datasets
* TILDA Fabric v2 Dataset (COCO format) (Available: https://tilda-dataset.org/fabric-v2)
### Dataset Classes
* Broken Stitch
* Needle Mark
* Pinched Fabric
* Vertical Defect
* Defect Free
* Hole
* Horizontal Defect
* Lines
* Objects
* Oil Spot
* Stain
* Thread Error
### Dataset Size
* Total Images: 11,572
* Training: 6,419
* Validation: 2,935
* Testing: 2,218


## Data Preprocessing
Preprocessing steps applied:
* TILDA COCO annotation conversion to classification format
* Image resizing: **128 × 128**
* Normalization: pixel values scaled to **[0,1]**
* Data augmentation:
  * Rotation
  * Zoom
  * Horizontal Flip
  * Vertical Flip
These preprocessing steps improved generalization and reduced overfitting.


## Model Architecture
The model uses a dual-backbone CNN architecture.
### Backbone 1: EfficientNetB3
Provides deep hierarchical feature extraction using compound scaling.
### Backbone 2: MobileNetV2
Captures lightweight and fine-grained texture patterns efficiently.
### Feature Fusion Pipeline
1. Input image fed into both backbones
2. Feature maps extracted independently
3. Global Average Pooling applied
4. Features concatenated
5. Dense classification layer
6. Softmax output for 12 classes


## Training Configuration
### Frameworks
* Python
* TensorFlow
* Keras
### Hyperparameters
* Batch Size: 32
* Epochs: 30
* Optimizer: AdamW
* Learning Rate: 1e-4
* Loss Function: Sparse Categorical Focal Loss
### Callbacks
* EarlyStopping
* ReduceLROnPlateau
* ModelCheckpoint


## Results
The model achieved strong classification performance across all defect classes.
### Performance Metrics
* Accuracy: **98.83%**
* Macro F1 Score: **0.97**
* Weighted F1 Score: **0.99**
### Evaluation Metrics
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC-AUC Curves


Sample outputs include:
* Confusion Matrix
* Classification Report
* ROC Curves
* Real-time prediction screenshots


## Backend / API
The trained model was deployed using **FastAPI**.
API capabilities:
Input:
* Fabric image upload
Output:
* Predicted defect class
* Confidence score
The backend handles:
* Image preprocessing
* Model inference
* Prediction response
This enables real-time defect detection for industrial usage.


## Frontend
A web-based frontend was developed to interact with the FastAPI backend.
Features:
* Fabric image upload
* Instant prediction
* Confidence score visualization
* User-friendly inspection interface
The frontend allows real-time testing of the AI model for production environments.


## Skills Demonstrated
* Deep Learning
* Convolutional Neural Networks (CNN)
* Transfer Learning
* Computer Vision
* Image Classification
* Data Augmentation
* Model Optimization
* FastAPI Deployment
* End-to-End ML System Development


## Research Publication
This research work is currently **under review in the Journal of Textile Association**.


## Project Note
The complete source code for the training pipeline, FastAPI backend, and frontend application is currently not included in this repository because the research paper associated with this project is under peer review for publication in the **Journal of Textile Association**.
To avoid issues related to unpublished research, intellectual property, and duplicate public disclosure during the review process, only the project documentation, model architecture details, evaluation metrics, and result visualizations are shared at this stage.
This repository currently includes:
* Project overview and methodology
* Model architecture description
* Experimental results and evaluation metrics
* Confusion matrix, ROC curve, classification report
* Sample real-world predictions
The complete implementation (training scripts, backend APIs, and frontend source code) may be made publicly available after the review and publication process is completed.
