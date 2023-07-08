# CIFAR-10 Object Detection with MobileNetV2

This project demonstrates object detection using the CIFAR-10 dataset and the MobileNetV2 model. It trains a custom model by adding layers on top of the pre-trained MobileNetV2 model and performs object detection on the CIFAR-10 dataset.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. Each image is labeled with one of the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The dataset is preprocessed by scaling the pixel values between 0 and 1 (float32) to normalize the data before training the model.

## Requirements

To run the project, you need to have the following dependencies installed:

- Python (version 3.6 or later)
- NumPy (```pip install numpy```)
- TensorFlow (```pip install tensorflow```)
- Matplotlib (```pip install matplotlib```)

## Getting Started

To get started with the project, follow these steps:

1. Clone the project repository from GitHub.
2. Install the required dependencies as mentioned in the "Requirements" section.
3. Open a terminal or command prompt and navigate to the project directory.
4. Run the script `object_detection.py` using the command: `python object_detection.py`.
5. The script will load the CIFAR-10 dataset, preprocess the data, create a custom model on top of the pre-trained MobileNetV2 model, and train the model.
6. During training, the accuracy and loss curves will be plotted using Matplotlib.
7. After training, the model will be evaluated on the test set, and the test accuracy will be displayed in the terminal.

## Model Architecture

The project uses the MobileNetV2 model pre-trained on the ImageNet dataset as the base model. Custom layers are added on top of the base model for object detection.

The custom layers include a GlobalAveragePooling2D layer to reduce the spatial dimensions, a Dense layer with 256 units and ReLU activation function, and a Dense layer with 10 units (output layer) and softmax activation function.

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function. The accuracy metric is used for evaluation.

## Results

The project plots the accuracy and loss curves during training using Matplotlib. The curves show the training and validation accuracy/loss over epochs, allowing you to analyze the model's performance.

After training, the model is evaluated on the test set, and the test accuracy is displayed in the terminal.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to modify and adapt the code according to your needs.

If you have any questions or suggestions, please feel free to contact me.

**Author:** Sajad Waloo
**Email:** sajadwaloo786@gmail.com

---
