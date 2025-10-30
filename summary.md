# Project Summary: Marathi Sentiment Analysis

This document provides a summary of the `main.py` and `main.ipynb` files, outlining the process of training a sentiment analysis model for Marathi text.

## `main.ipynb`: Data Preprocessing

This Jupyter Notebook is responsible for the initial data preprocessing.

- **Loads Data**: It starts by loading the raw training, validation, and test datasets from CSV files located in `dataset/raw/`.
- **Combines and Cleans**: The separate datasets are concatenated into a single DataFrame. Duplicate text entries are removed to prevent data leakage, and the entire dataset is shuffled to ensure randomness.
- **Splits Data**: The cleaned dataset is then split back into training, validation, and test sets with specific sizes.
- **Saves Preprocessed Data**: Finally, these new datasets are saved as CSV files in the `dataset/preprocess/` directory, ready for use in the training pipeline.

## `main.py`: Model Training and Evaluation

This Python script handles the end-to-end process of training and evaluating the sentiment analysis model.

- **`BiLSTM` Class**: A custom class that encapsulates all the logic for the model.
  - **Tokenization**: It uses a `Tokenizer` to convert the text data into sequences of integers. The tokenizer is fitted on the training data and then used to transform all datasets.
  - **Label Mapping**: The sentiment labels `{-1, 0, 1}` are mapped to `{0, 1, 2}` for compatibility with the model's output layer.
  - **Model Architecture**: A Bidirectional LSTM (BiLSTM) model is built using Keras. The architecture consists of an `Embedding` layer, `SpatialDropout1D`, a `Bidirectional LSTM` layer, and a `Dense` output layer with a softmax activation function.
  - **Training**: The model is trained using the preprocessed data. It employs several callbacks to improve the training process:
    - `EarlyStopping`: To prevent overfitting by stopping the training when the validation loss stops improving.
    - `ReduceLROnPlateau`: To adjust the learning rate during training.
    - `ModelCheckpoint`: To save the best version of the model based on validation accuracy.
  - **Evaluation**: After training, the model is evaluated on the test set. It prints a classification report and a confusion matrix to assess the model's performance.
  - **Plotting**: The script generates and saves plots for training/validation accuracy and loss, as well as a heatmap of the confusion matrix.
- **`main()` function**: This function orchestrates the entire process, from loading the preprocessed data to training, evaluating, and saving the final model and tokenizer.

## Key Points for Teammates

When explaining this project, here are some important points to highlight:

- **Data Pipeline**: Emphasize the two-step process: data preprocessing in the notebook and model training in the Python script. This separation keeps the concerns of data preparation and modeling distinct.
- **Data Cleaning**: Mention the removal of duplicates and shuffling of the data as crucial steps for building a robust model.
- **Model Choice**: Explain why a BiLSTM was chosen. It's well-suited for sequence data like text because it can capture context from both past and future words in a sentence.
- **Callbacks**: Discuss the importance of the Keras callbacks used during training. They help in regularizing the model and finding the best set of weights.
- **Artifacts**: Point out that the script saves not only the trained model (`.keras` file) but also the `Tokenizer` (`.pkl` file) and performance plots (`.png` files). The tokenizer is essential for making predictions on new, unseen text.
- **Label Mapping**: Explain the necessity of mapping the original `{-1, 0, 1}` labels to `{0, 1, 2}` to work with `SparseCategoricalCrossentropy` loss.
