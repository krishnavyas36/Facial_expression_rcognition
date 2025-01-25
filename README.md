# Facial Expression Recognition  

## Overview  
This project focuses on detecting facial expressions using both Machine Learning (ML) and Deep Learning (DL) techniques. The goal is to classify images of faces into various emotion categories, such as happy, sad, angry, etc. The project utilizes the [FER (Facial Expression Recognition) dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?select=train) for training and evaluation.  

## Dataset  
The dataset consists of facial images categorized into multiple emotion labels. The dataset is divided into training and testing sets, providing a solid foundation for building and evaluating models.  
- **Source**: [FER Dataset on Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?select=train)  
- **Emotion Categories**: Examples include Happy, Sad, Angry, Surprise, Neutral, etc.  

## Features  
- Implemented **Machine Learning Models** for initial prototyping and baseline accuracy.  
- Developed **Deep Learning Models** using Convolutional Neural Networks (CNNs) for improved performance.  
- Data preprocessing includes normalization, resizing, and augmenting facial images.  
- Training and evaluation metrics, such as accuracy, confusion matrix, and F1-score, to measure performance.  

## Technologies Used  

### Machine Learning  
- **Scikit-Learn**: For building and evaluating traditional ML models such as SVM, Random Forest, and KNN.  

### Deep Learning  
- **TensorFlow**: For designing, training, and optimizing CNN models.    

### Other Tools  
- **Matplotlib/Seaborn**: For visualizing data and model performance.  

## Setup  

### Prerequisites  
- **Python**: Version 3.7 or higher.  
- Install essential Python libraries:  
  ```bash  
  pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python  
  ```  

### Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/krishnavyas36/Facial_expression_recognition.git  
   cd Facial_expression_recognition  
   ```  

2. Download the FER dataset from [Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?select=train).  
   - Place the dataset files in a directory named `data/` within the project folder.  

3. Run the preprocessing script to prepare the dataset:  
   ```bash  
   python preprocess.py  
   ```  

4. Train the ML model:  
   ```bash  
   python train_ml.py  
   ```  

5. Train the DL model:  
   ```bash  
   python train_dl.py  
   ```  

## Usage  

- **Training**:  
  Modify hyperparameters in the respective scripts (`train_ml.py` or `train_dl.py`) as needed, and run the training scripts to build models.  

- **Evaluation**:  
  Use the evaluation script to test the performance of trained models on the test set:  
  ```bash  
  python evaluate.py  
  ```  

- **Prediction**:  
  Use the prediction script to classify a new image:  
  ```bash  
  python predict.py --image_path <path_to_image>  
  ```  

## Results  
- **Machine Learning Models**:  
  - Achieved baseline accuracy using models like Support Vector Machines (SVM) and Random Forest.  

- **Deep Learning Models**:  
  - CNN-based models demonstrated significant performance improvement with better generalization on unseen data.  

### Sample Output  
Include sample accuracy scores, confusion matrices, or training curves for both ML and DL models here.  

## Contributing  
Contributions to this project are welcome! If you have suggestions, please create an issue or submit a pull request.  

## License  
This project is for educational purposes. Please provide appropriate attribution if you use this work.  

