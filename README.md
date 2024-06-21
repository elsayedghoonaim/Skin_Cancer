## Skin Cancer Classification Project

This project focuses on classifying skin lesions as either benign or malignant using a Convolutional Neural Network (CNN) model.

### Project Structure

* **data/train:** Contains training images categorized into 'benign' and 'malignant' folders.
* **data/test:** Contains testing images categorized into 'benign' and 'malignant' folders.

### Dependencies

* TensorFlow (with GPU support recommended)
* Keras
* OpenCV (cv2)
* NumPy
* scikit-learn

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required libraries:
   ```bash
   pip install tensorflow keras opencv-python numpy scikit-learn
   ```

### Usage

1. **Data Preparation:**
   * Organize your skin lesion images into the 'train' and 'test' directories, each with subdirectories for 'benign' and 'malignant' classes.
   * Ensure images are in a compatible format (e.g., JPG, PNG).

2. **Model Training:**
   * Run the provided Python script to train the CNN model. The script will:
      * Load and preprocess the training images.
      * Define and compile the CNN model.
      * Train the model for 60 epochs.

3. **Model Evaluation:**
   * The script will automatically evaluate the model's performance on the test dataset and print the test accuracy.

### Model Architecture

* The CNN model consists of multiple convolutional layers, max-pooling layers, and dropout layers for regularization.
* The final layer uses a sigmoid activation function to produce binary classification output.

### Results

* Test accuracy achieved: 84.09%

### Improvements

* **Data Augmentation:** Apply data augmentation techniques (e.g., rotation, flipping) to increase the diversity of training data.
* **Hyperparameter Tuning:** Experiment with different hyperparameters (e.g., learning rate, batch size) to optimize the model's performance.
* **Model Architecture:** Explore alternative CNN architectures or transfer learning to potentially improve accuracy.

### Additional Notes

* The code assumes that you have GPU resources available (with tf.device('/gpu:0')). If not, remove or modify the device specifications.
* Consider using a larger and more diverse dataset for better generalization.

Feel free to contribute to this project by improving the model, adding new features, or exploring different approaches to skin cancer classification.

**Disclaimer:** This project is for educational and research purposes only and should not be used for medical diagnosis. Always consult a qualified healthcare professional for any medical concerns.

Let me know if you have any other questions. 
