# Introduction-to-deep-learning

Predictive Maintenance using Machine Sound
This project aims to build a predictive maintenance model using the sound of machines. The model is trained to classify whether a specific machine (Machine 9) is running or not based on the machine's sound. This is a binary classification problem where '1' indicates that the machine is on and '0' indicates that the machine is off.

Project Structure
The project consists of a Jupyter notebook (Classifier2.ipynb) which contains all the code required to train and evaluate the model.

The notebook is structured as follows:

Importing necessary libraries: The necessary Python libraries for data handling, audio processing, and machine learning are imported.

Loading the data: The audio files are loaded from a specified directory. Each audio file corresponds to a recording of the machine's sound.

Generating the features (MFCCs): The Mel-frequency cepstral coefficients (MFCCs) of the audio files are computed using the librosa library. MFCCs are commonly used features in audio and speech processing.

Label extraction: The labels indicating whether the machine is on or off are extracted directly from the audio filenames.

Data preparation: The MFCCs and labels are split into training and testing sets.

Model training: A neural network is trained on the training set. The model's architecture consists of a flatten layer, a dense layer with ReLU activation, a dropout layer for regularization, and a final dense layer with sigmoid activation for binary classification.

Model evaluation: The trained model is evaluated on the testing set.

Prerequisites
The project uses the following Python libraries:

os
pandas
numpy
librosa
matplotlib.pyplot
sklearn.model_selection
tensorflow.keras.models
tensorflow.keras.layers
tensorflow.keras.optimizers
These libraries can be installed via pip:

bashCopy codepip install pandas numpy librosa matplotlib scikit-learn tensorflow

UsageTo use the model, follow the steps below:
Clone this repository to your local machine.
Install the necessary Python libraries if not already installed.Run the Jupyter notebook (Classifier2.ipynb).
The notebook is self-contained and should run without any issues provided that the necessary Python libraries are installed and the audio files are in the correct directory.
Contributing
Contributions to this project are welcome. If you find a bug or think of a feature that would benefit the project, please open an issue or submit a pull request.
LicenseThis project is licensed under the MIT License. See the LICENSE file for details.
