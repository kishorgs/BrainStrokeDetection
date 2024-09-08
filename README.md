Here's an updated version of your README file, including a section for an output demo video at the beginning:

---

# Brain Stroke Detection Using CNN and CT Scan Images

## Demo Video

Watch the demo video to see the Brain Stroke Detection model in action:

[![Demo Video](img/Demo.mp4)]

This video showcases the functionality of the Tkinter-based GUI interface for uploading CT scan images and receiving predictions on whether the image indicates a brain stroke or not.

## Project Overview

This project focuses on detecting brain strokes using machine learning techniques, specifically a Convolutional Neural Network (CNN) algorithm. The model is trained on a dataset of CT scan images to classify images as either "Stroke" or "No Stroke". The dataset was sourced from Kaggle, and the project uses TensorFlow for model development and Tkinter for a user-friendly interface.

### Key Features:
- **Machine Learning Model**: CNN model built using TensorFlow for classifying brain stroke based on CT scan images.
- **User Interface**: Tkinter-based GUI for easy image uploading and prediction.
- **Visualization**: Includes model performance metrics such as accuracy, ROC curve, PR curve, and confusion matrix.

### Dataset:
- **Source**: [Kaggle - Brain Stroke CT Image Dataset](https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset)
- **Classes**: The dataset contains two categories:
  - **Normal** (No Stroke)
  - **Stroke** (Brain Stroke)

## Project Structure

### Model Architecture

The CNN model architecture consists of:
- 3 Convolutional layers with ReLU activation and MaxPooling.
- 2 Dense layers of 500 units with ReLU activation.
- Dropout layers (20%) for regularization.
- A final dense layer with a sigmoid activation function for binary classification.

**Training**: The model was trained on images resized to 224x224 pixels, normalized for optimal learning.

### Evaluation Metrics:

- **Accuracy on Test Data**: 90%+
- **ROC and PR Curves**: Graphical metrics to evaluate the model's performance.
- **Loss and Accuracy**: Tracked during training for both training and validation sets.

## How to Use the Project

### Requirements

- **Python 3.8+**
- **TensorFlow 2.17.0**
- **Tkinter**
- **PIL (Pillow)**

Install dependencies via `pip`:
```bash
pip install tensorflow pillow matplotlib scikit-learn
```

### Running the Project

1. **Training the Model**:
   - The project code automatically splits the dataset and trains the model.
   - The model is saved as `stroke_detection_model.h5` after training.

2. **Using the Tkinter Interface**:
   - Run the interface using the provided Tkinter code.
   - Upload any CT scan image, and the interface will predict whether the image shows signs of a brain stroke.

```bash
python stroke_detection_app.py
```

### Screenshots

**Main Interface**:
- The application allows you to upload a CT scan image and provides the prediction.

**Output Screens**:
- Predictions will show either "Stroke" or "No Stroke" based on the uploaded image.

**Model Summary**:
- A detailed architecture overview of the CNN model.

**Training Epochs**:
- Visual representation of accuracy and loss during model training.

**Performance Metrics**:
- ROC and PR curve graphs to evaluate the performance of the model.

## Results

- The CNN model achieves over 90% accuracy on the test data.
- **ROC Curve**: The model shows good discriminative ability in predicting strokes.
- **PR Curve**: Precision and recall metrics indicate strong performance, especially for stroke detection.
  
### Confusion Matrix:
- A confusion matrix indicates how well the model classifies stroke and non-stroke cases.

## Conclusion

This project successfully implements a machine learning model for detecting brain strokes using CT scan images. The developed GUI allows easy image uploading and prediction, enhancing accessibility for healthcare applications. Further improvements could involve testing the model on a larger dataset and optimizing the architecture.

---

Replace `"your-video-id"` with the actual YouTube video ID for your demo video. This README provides a comprehensive overview and showcases the functionality of your project effectively.