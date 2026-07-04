# Toxic Comment Classification System

A deep learning-based system to automatically detect and classify toxic comments on online platforms. The model identifies six toxicity subtypes—**toxic**, **severe_toxic**, **obscene**, **threat**, **insult**, and **identity_hate**—to support content moderation and community safety initiatives.

---

## 📌 Project Overview

Online platforms such as Wikipedia, social media, and forums generate massive volumes of user comments daily. While many comments are constructive, a significant portion can be harmful, abusive, or toxic. Manual moderation is slow and expensive, while simple keyword-based filters lack context and often miss nuanced toxicity.

This project addresses these challenges by building a **multi-label deep learning classifier** that:
- Automatically detects whether a comment is toxic.
- Classifies the comment into one or more toxicity categories.
- Provides probability scores for each category to support nuanced moderation decisions.

The system is implemented using **TensorFlow/Keras** with a **Bidirectional LSTM** architecture and is deployed as an interactive web application using **Gradio**.

---

## 🎯 Problem Statement

- **Scale**: Platforms receive thousands to millions of comments daily, making manual review impractical.
- **Nuance**: Toxicity can be subtle, context-dependent, and expressed in varied linguistic styles.
- **Multi-label nature**: A single comment can be toxic, obscene, and an insult simultaneously.
- **Bias & fairness**: Models must avoid over-flagging marginalized language while still catching genuine harm.

This project aims to build a robust, scalable, and interpretable model to assist in automated content moderation.

---

## 📊 Dataset

This project uses the **Jigsaw Toxic Comment Classification dataset** (Wikipedia comments):
- **Source**: [Hugging Face - google/jigsaw_toxicity_pred](https://huggingface.co/datasets/google/jigsaw_toxicity_pred) [web:16]
- **Size**: 159,571 comments
- **Labels** (binary for each comment):
  - `toxic`
  - `severe_toxic`
  - `obscene`
  - `threat`
  - `insult`
  - `identity_hate`

### Data Distribution

- **Clean comments** (no labels): ~143,346 (~90%)
- **Comments with at least one label**: ~16,225 (~10%)

This strong imbalance (many more non-toxic than toxic comments) requires careful model design and evaluation.

---

## 🧠 Model Architecture

The model is built using **TensorFlow 2.15** and **Keras** with the following architecture:

```text
Input (Text)
  ↓
Tokenization & Padding (200,000 features, max length 300)
  ↓
Embedding Layer (128-dimensional)
  ↓
Bidirectional LSTM (64 units, return_sequences=True)
  ↓
Dropout (0.5)
  ↓
Bidirectional LSTM (64 units)
  ↓
Dropout (0.5)
  ↓
Dense (128 units, ReLU)
  ↓
Dropout (0.5)
  ↓
Dense (6 units, Sigmoid)  ← Multi-label output
```

### Key Design Choices

- **Bidirectional LSTM**: Captures context from both past and future tokens, improving understanding of sentence structure and meaning.
- **Dropout layers**: Reduce overfitting, especially important due to class imbalance and limited toxic samples.
- **Sigmoid output + binary_crossentropy**: Enables multi-label classification where each toxicity class is predicted independently.
- **Padding & Tokenization**: Uniform sequence length (300) ensures consistent input shapes and efficient batch processing.

---

## 🛠️ Tech Stack & Tools

- **Language**: Python 3.10
- **Deep Learning**: TensorFlow 2.15, Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Text Preprocessing**: NLTK (stopwords), Regular Expressions
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Gradio (interactive web UI)
- **Environment**: Google Colab, Google Drive

---

## 📁 Project Structure

```text
/toxic-comment-classifier
│
├── train.csv                 # Raw dataset (Jigsaw Toxic Comment Classification)
├── preprocessing.ipynb       # (Optional) Data exploration & preprocessing notebook
├── model_training.ipynb      # Main training script (this repo's core)
├── app.py                    # Gradio-based web interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
```

You can adapt file names as needed to match your actual project layout.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/toxic-comment-classifier.git
cd toxic-comment-classifier
```

### 2. Create & Activate a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
tensorflow==2.15
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
gradio
```

### 4. Download / Prepare the Dataset

Place the dataset file (`train.csv`) in the project root or in a `data/` folder:

```text
/data
  └── train.csv
```

Update the data path in the training script accordingly.

---

## 🧪 Training the Model

The main training logic is implemented in `model_training.ipynb` (or a `.py` script). Key steps:

1. **Load and preprocess data**:
   - Lowercase text.
   - Remove non-alphanumeric characters.
   - Remove stopwords using NLTK.

2. **Tokenize and pad sequences**:
   - Use `Tokenizer` with 200,000 features.
   - Pad sequences to length 300.

3. **Split data**:
   - Train: 70%
   - Validation: 30% of original 30% (≈21%)
   - Test: 9% of original 30% (≈9%)

4. **Build and train model**:
   - Compile with `binary_crossentropy` loss and `adam` optimizer.
   - Train for 10 epochs with batch size 32.
   - Use `tf.data.Dataset` for efficient batching and prefetching.

Example training snippet (from your code):

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)
```

Expected results (based on your experiment):
- **Test accuracy**: ~97.9%
- **Validation accuracy**: ~97–99% across epochs

---

## 🌐 Running the Web Interface

The project includes a **Gradio-based web app** that allows users to:
- Input a comment.
- Receive toxicity predictions for all six categories.

### 1. Run the App Locally

```bash
python app.py
```

`app.py` defines a function like:

```python
def score_comment_interface(comment):
    result = score_comment(comment)
    result_text = ""
    for idx, col in enumerate(df.columns[2:]):
        result_text += f"{col}: {'yes' if result[idx] > 0.5 else 'No'}\n"
    return result_text
```

### 2. Deploy to Gradio (Cloud)

From Google Colab or local environment:

```python
import gradio as gr

interface = gr.Interface(
    fn=score_comment_interface,
    inputs=gr.Textbox(lines=2, placeholder='Comment to score'),
    outputs='text'
)
interface.launch(share=True)
```

This generates a public URL (expires in 72 hours by default) for testing and sharing.

---

## 📈 Evaluation & Results

### Metrics

- **Accuracy**: Overall correctness across all samples.
- **Per-class behavior**: Implicitly captured via sigmoid outputs; thresholded at 0.5 for binary decisions.

### Observed Performance

- **Training accuracy**: Increased from ~93.5% to ~98.9% over 10 epochs.
- **Validation accuracy**: Stable around 97–99%.
- **Test accuracy**: ~97.94%.

While overall accuracy is high, note that:
- The dataset is heavily imbalanced (most comments are clean).
- Additional metrics (e.g., F1-score per class, false positives/negatives) would give deeper insight into performance on toxic subsets.

---

## 🔍 Limitations & Future Work

### Limitations

- **Class imbalance**: The model may be biased toward predicting “non-toxic” due to the large number of clean comments.
- **Context limitations**: 300-token max length may truncate very long comments.
- **Preprocessing**: Simple stopword removal may discard some contextually important words.
- **Evaluation**: Primarily accuracy-based; no per-class F1 or ROC analysis implemented.

### Future Enhancements

- Use **class weighting** or **focal loss** to handle imbalance.
- Experiment with:
  - **BERT** or other transformer-based embeddings.
  - **CNN + LSTM** or **Transformer** architectures.
- Add:
  - Per-class F1-score, precision, and recall.
  - Confusion matrices for each toxicity label.
- Deploy as:
  - A REST API using **FastAPI**.
  - A containerized service (Docker) for integration into production moderation systems.

---

## 📚 References

- Jigsaw Toxic Comment Classification Challenge dataset:  
  [Hugging Face - google/jigsaw_toxicity_pred](https://huggingface.co/datasets/google/jigsaw_toxicity_pred) [web:16]
- TensorFlow Datasets: `wikipedia_toxicity_subtypes`  
  [TensorFlow Datasets - wikipedia_toxicity_subtypes](https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes) [web:19]
- Related research on LSTM/BiLSTM for toxicity detection:  
  - “An Automated Toxicity Classification on Social Media Using LSTM and Word Embedding” [web:14]  
  - “Bidirectional LSTM with Convolution for Toxic Comment Classification” [web:11][web:12]

---

## 🤝 Contributing

Contributions are welcome! If you’d like to improve the model, add features, or fix issues:

1. Fork the repository.
2. Create a new branch:  
   `git checkout -b feature/<your-feature>`
3. Make your changes and commit:  
   `git commit -m "Add: <description>"`
4. Push and open a pull request.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file (if added) for details.

---

## 👤 Author

##Dyuti Oindrela## 
Hong Kong, HK  
GitHub: [@<your-username>](https://github.com/<your-username>)

---

## 🙏 Acknowledgments

- Jigsaw and Google for releasing the Toxic Comment Classification dataset.
- TensorFlow and Keras teams for robust deep learning tools.
- Gradio for enabling rapid deployment of ML demos.
