# CNN
This code demonstrates the implementation of a Convolutional Neural Network (CNN) using the PyTorch framework for the task of medical image classification.
Here's a breakdown of the code:

Model Definition (MedicalCNN class):

The CNN model is defined with two convolutional layers (conv1 and conv2), ReLU activation functions (relu1, relu2), and max-pooling layers (pool1, pool2).
The flattened output is connected to two fully connected layers (fc1 and fc2) for classification.
Data Loading and Preprocessing:

The code uses the MNIST dataset for demonstration purposes. It loads the dataset using PyTorch's DataLoader and applies a transformation to convert images to tensors.
Initialization of Model, Loss Function, and Optimizer:

The CNN model (MedicalCNN) is instantiated, and the CrossEntropyLoss is used as the loss function.
The Adam optimizer is employed to update the model parameters during training.
Training Loop:

The script enters a training loop that iterates through a specified number of epochs (5 in this case).
The model is set to training mode (model.train()), and for each batch in the training dataset, it computes predictions, calculates the loss, and updates the model parameters using backpropagation.
After each epoch, the model is evaluated on the test set to measure its accuracy. The model is set to evaluation mode (model.eval()), and accuracy is calculated.
Device Configuration:

The code checks whether a GPU (CUDA) is available and moves the model to the GPU if possible. This improves training speed if a GPU is present; otherwise, it uses the CPU.
Model Saving:

After training, the script saves the trained model's parameters to a file named "medical_cnn_model.pth" using torch.save().
Overall, this code serves as a basic template for training a CNN on a medical image classification task using PyTorch. You can adapt it to your specific dataset and task by modifying the data loading part and adjusting the model architecture accordingly.
