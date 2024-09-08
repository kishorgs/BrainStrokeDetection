<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Stroke Detection Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        .container {
            max-width: 1200px;
            margin: auto;
        }
        video {
            width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Brain Stroke Detection Using CNN and CT Scan Images</h1>

        <h2>Demo Video</h2>
        <video controls autoplay>
            <source src="path/to/your/local/video.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>

        <h2>Project Overview</h2>
        <p>This project focuses on detecting brain strokes using machine learning techniques, specifically a Convolutional Neural Network (CNN) algorithm. The model is trained on a dataset of CT scan images to classify images as either "Stroke" or "No Stroke". The dataset was sourced from Kaggle, and the project uses TensorFlow for model development and Tkinter for a user-friendly interface.</p>

        <h3>Key Features:</h3>
        <ul>
            <li>Machine Learning Model: CNN model built using TensorFlow for classifying brain stroke based on CT scan images.</li>
            <li>User Interface: Tkinter-based GUI for easy image uploading and prediction.</li>
            <li>Visualization: Includes model performance metrics such as accuracy, ROC curve, PR curve, and confusion matrix.</li>
        </ul>

        <h3>Dataset:</h3>
        <p>Source: <a href="https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset" target="_blank">Kaggle - Brain Stroke CT Image Dataset</a></p>
        <p>Classes: The dataset contains two categories:
            <ul>
                <li>Normal (No Stroke)</li>
                <li>Stroke (Brain Stroke)</li>
            </ul>
        </p>

        <h2>Project Structure</h2>

        <h3>Model Architecture</h3>
        <p>The CNN model architecture consists of:
            <ul>
                <li>3 Convolutional layers with ReLU activation and MaxPooling.</li>
                <li>2 Dense layers of 500 units with ReLU activation.</li>
                <li>Dropout layers (20%) for regularization.</li>
                <li>A final dense layer with a sigmoid activation function for binary classification.</li>
            </ul>
        </p>

        <p><strong>Training</strong>: The model was trained on images resized to 224x224 pixels, normalized for optimal learning.</p>

        <h3>Evaluation Metrics:</h3>
        <ul>
            <li>Accuracy on Test Data: 90%+</li>
            <li>ROC and PR Curves: Graphical metrics to evaluate the model's performance.</li>
            <li>Loss and Accuracy: Tracked during training for both training and validation sets.</li>
        </ul>

        <h2>How to Use the Project</h2>

        <h3>Requirements</h3>
        <p>Python 3.8+</p>
        <p>TensorFlow 2.17.0</p>
        <p>Tkinter</p>
        <p>PIL (Pillow)</p>

        <p>Install dependencies via <code>pip</code>:</p>
        <pre><code>pip install tensorflow pillow matplotlib scikit-learn</code></pre>

        <h3>Running the Project</h3>
        <ol>
            <li><strong>Training the Model</strong>: The project code automatically splits the dataset and trains the model. The model is saved as <code>stroke_detection_model.h5</code> after training.</li>
            <li><strong>Using the Tkinter Interface</strong>: Run the interface using the provided Tkinter code. Upload any CT scan image, and the interface will predict whether the image shows signs of a brain stroke or not.</li>
        </ol>
        <pre><code>python stroke_detection_app.py</code></pre>

        <h3>Screenshots</h3>
        <p>Main Interface: The application allows you to upload a CT scan image and provides the prediction.</p>
        <p>Output Screens: Predictions will show either "Stroke" or "No Stroke" based on the uploaded image.</p>
        <p>Model Summary: A detailed architecture overview of the CNN model.</p>
        <p>Training Epochs: Visual representation of accuracy and loss during model training.</p>
        <p>Performance Metrics: ROC and PR curve graphs to evaluate the performance of the model.</p>

        <h2>Results</h2>
        <p>The CNN model achieves over 90% accuracy on the test data.</p>
        <p><strong>ROC Curve</strong>: The model shows good discriminative ability in predicting strokes.</p>
        <p><strong>PR Curve</strong>: Precision and recall metrics indicate strong performance, especially for stroke detection.</p>

        <h3>Confusion Matrix</h3>
        <p>A confusion matrix indicates how well the model classifies stroke and non-stroke cases.</p>

        <h2>Conclusion</h2>
        <p>This project successfully implements a machine learning model for detecting brain strokes using CT scan images. The developed GUI allows easy image uploading and prediction, enhancing accessibility for healthcare applications. Further improvements could involve testing the model on a larger dataset and optimizing the architecture.</p>
    </div>
</body>
</html>
