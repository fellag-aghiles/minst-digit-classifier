MNIST Digit Recognition with CNN
================================

This project lets you recognize handwritten digits using a simple Convolutional Neural Network (CNN) built with PyTorch. 
You can either draw a digit or select an image, and the program will predict what number it is.

Features
--------
- Train a CNN on the MNIST dataset (80% training / 20% validation)
- Use a pretrained model or train a new one yourself
- Automatically uses GPU if available, otherwise CPU
- Test your own images with a simple file picker
- See predictions both in the console and as an image pop-up

How the Model Works
------------------
- 3 convolutional layers with ReLU activation
- 1 fully connected layer with 10 outputs (digits 0–9)
- ADAM optimizer
- Cross-entropy loss
- Trained for 4 epochs to avoid overfitting

Training Details
----------------
- Dataset: MNIST
- Training Split: 80%
- Validation Split: 20%
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 4
- Batch Size: 32
- Device: GPU (CUDA) if available, otherwise CPU


How to Use
----------
Run the program:
   python mnist_cnn.py

You’ll be asked:
   Do you want to train the model? (y/n):

- Type 'y' to train the model from scratch (this may take a few minutes)
- Type 'n' to load the pretrained model (if `model_state.pt` exists)

Next, (or after trinning if you click y) a file dialog will appear so you can pick an image of a minst digit. 
The program will automatically preprocess it and show the predicted number.

Example Output
--------------
Epoch:3 train_loss=0.0621 val_loss=0.0457 val_acc=0.9874
Test loss: 0.0432 Test accuracy: 0.9881
Model prediction: 4
