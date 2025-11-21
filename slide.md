# Slide 1: Title

**Title: Marathi Movie Sentiment Analysis**
**Subtitle: A Deep Learning Approach**
**Presented by:**
    - Kanishk Richhariya: 23103125
    - Kartik Agrawal: 23803013
    - Vihan Tandon: 23803015
    - Hansi Gupta: 23803019
    - Aditya Pandey: 23803024

---

## Slide 2: Introduction

* **What is Sentiment Analysis?**
* The process of computationally identifying and categorizing opinions expressed in a piece of text.
* Used to determine whether the writer's attitude towards a particular topic, product, etc., is positive, negative, or neutral.
* **Why Marathi?**
* Marathi is a major regional language in India with millions of speakers.
* There is a growing need for NLP tools for regional languages to power applications, services, and content analysis.
* **Project Goal**
* To develop a deep learning model capable of accurately classifying Marathi Movie reviews into Positive, Negative, and Neutral sentiments.

---

### Slide 3: The Dataset

* **Dataset:** MahaSent
* **Source:** Raw data from `https://github.com/l3cube-pune/MarathiNLP/tree/main/L3Cube-MahaSent-MD`.
* **Labels:** The dataset uses a three-class sentiment system:
    -`1`: Positive
    -`0`: Neutral
    -`-1`: Negative
* **Initial State:** The notebook `main.ipynb` loads these files, combines them, and performs initial cleaning and preparation.

---

### Slide 4: Data Preprocessing

* **Cleaning:**
    -All datasets (`train`, `val`, `test`) were combined into a single DataFrame.
    -Duplicate text entries were removed to prevent data leakage and model bias.
    -The entire dataset was shuffled to ensure randomness and unbiased splits.
* **Splitting:**
    -The cleaned data was strategically split into new sets for robust training and evaluation:
* **Training Set:** 50,509 samples
* **Validation Set:** 4,000 samples
* **Test Set:** 6,000 samples
* **Output:** The final, preprocessed datasets are saved as clean CSV files in the `dataset/preprocess/` directory.

---

### Slide 5: Model Architecture - BiLSTM

* **Why Bidirectional LSTM?**
    -LSTMs (Long Short-Term Memory networks) are a type of RNN, excellent for sequence data like text.
* **"Bidirectional"** means the model reads the text from both left-to-right and right-to-left. This allows it to capture context from both past and future words, leading to a deeper understanding of the sentence.
* **Core Layers:**

1. `Embedding`: Converts words into dense vectors of a fixed size (128 dimensions).
2. `SpatialDropout1D`: A special dropout layer that helps prevent overfitting in NLP models.
3. `Bidirectional(LSTM)`: The main processing layer with 96 units, learning from both directions of the text.
4. `Dense`: The final output layer with 3 units (for 3 classes) and a `softmax` activation to output class probabilities.

---

### Slide 6: Tokenization & Label Engineering

* **Tokenization:**
    -The `Tokenizer` from Keras was used to convert the raw Marathi text into sequences of integers.
    -A vocabulary of the top **10,000** most frequent words was built from the training data.
    -All sentences were padded or truncated to a uniform length of **125 words**.
* **Label Mapping:**
    -The model's loss function (`SparseCategoricalCrossentropy`) requires integer labels starting from 0.
    -The original labels `{-1, 0, 1}` were mapped to a zero-indexed format:
    -`Negative (-1) -> 0`
    -`Neutral (0)   -> 1`
    -`Positive (1)   -> 2`

---

### Slide 7: The Training Process

* **Compiler:**
    -**Optimizer:** `Adam` with an initial learning rate of `0.0005`.
    -**Loss Function:** `SparseCategoricalCrossentropy`, suitable for multi-class classification with integer labels.
    -**Smart Training with Callbacks:**
        -`ModelCheckpoint`: Saves the best version of the model based on `val_accuracy` after each epoch.
        -`EarlyStopping`: Stops training if `val_loss` doesn't improve for 7 consecutive epochs, preventing overfitting and saving time.
        -`ReduceLROnPlateau`: Reduces the learning rate if `val_loss` plateaus, helping the model navigate the loss landscape more effectively.
* **Configuration:**
    -**Epochs:** Set to 50 (but likely to stop early).
    -**Batch Size:** 64.

---

### Slide 8: Training Performance

*(Image: `history.png` would be displayed here)*

* **Accuracy:** The model's accuracy on both the training and validation sets increased steadily, indicating effective learning. The validation accuracy closely tracks the training accuracy, which suggests the model is generalizing well to unseen data.
* **Loss:** The training and validation loss decreased consistently. The minimal gap between the two curves demonstrates that our regularization techniques (Dropout, Early Stopping) were successful in preventing significant overfitting.

---

### Slide 9: Evaluation on Test Data

*(Image: `confusion_matrix.png` would be displayed here)*

* **Confusion Matrix:**
    -The diagonal shows a high number of correct predictions for each class.
    -The model is highly effective at identifying **Negative** and **Positive** sentiments.
    -As is common in sentiment analysis, the **Neutral** class is slightly more challenging to classify, with some confusion with negative and positive classes.
* **Classification Report:**
    -The model achieved a high overall **Test Accuracy of ~85%**.
* **Precision, Recall, and F1-score** are strong across all classes, particularly for the Positive and Negative categories, confirming the model's robust performance.

---

### Slide 10: Key Learnings

* **Robust Data Pipeline:** Emphasize the importance of a well-defined two-step process: data preprocessing in `main.ipynb` and model training in `main.py`. This separation ensures data integrity and modularity.
* **BiLSTM's Strength:** The Bidirectional LSTM proved effective in capturing contextual information from Marathi text, which is crucial for accurate sentiment classification.
* **Callback Efficacy:** Keras callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`) were instrumental in optimizing the training process, preventing overfitting, and saving the best model.
* **Challenges with Neutral Sentiment:** Classifying neutral sentiment often presents a greater challenge compared to distinctly positive or negative sentiments, a common observation in sentiment analysis tasks.
* **Artifact Management:** Saving the trained model (`.keras`) and the `Tokenizer` (`.pkl`) is vital for future inference and reproducibility.

---

### Slide 11: Future Directions

* **Experiment with other architectures:** Explore transformer-based models (e.g., mBERT, IndicBERT) for potentially higher accuracy and state-of-the-art performance.
* **Hyperparameter Tuning:** Systematically search for the optimal combination of parameters (e.g., embedding size, LSTM units, dropout rate) to further enhance model performance.
* **Data Augmentation:** Implement techniques like back-translation or synonym replacement to increase the size and diversity of the training data, especially for underrepresented classes.
* **Deployment:** Package the trained model and tokenizer into a REST API for seamless integration into real-world applications and services.

---

### Slide 12: Thank You

* Thank you!
