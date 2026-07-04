# Deep-Learning-Projects
Toxic Comment Classification System Using Bidirectional LSTM
Project Overview
Designed and implemented a deep learning-based system to automatically detect and classify toxic comments on online platforms. The model identifies six toxicity subtypes—toxic, severe_toxic, obscene, threat, insult, and identity_hate—to support content moderation and community safety initiatives.

Data & Preprocessing

Used the 
Jigsaw Toxic Comment Classification dataset
 containing 159,571 Wikipedia comments labeled for toxicity subtypes.

Performed text preprocessing including lowercasing, removal of non-alphanumeric characters, and filtering with NLTK English stopwords to reduce noise and improve model generalization.

Analyzed label distributions and identified that ~90% of comments were clean, highlighting the need for robust multi-label classification.

Model Architecture

Built a multi-label classification model using TensorFlow/Keras with:

A TextVectorization/Tokenizer layer with 200,000 features and 300-token sequences.

An embedding layer (128-dimensional) followed by two stacked Bidirectional LSTM layers (64 units each) with dropout (0.5) for regularization.

Dense layers (128 → 6) with sigmoid activations to output probabilities for each toxicity class.

The model achieved ~97.9% test accuracy and maintained high validation accuracy (~97–99%) over 10 training epochs, demonstrating effective learning despite label imbalance.

Deployment

Developed an interactive web interface using Gradio that allows users to input comments and receive real-time toxicity predictions for all six categories.

Deployed the model as a publicly accessible application with a shared URL, enabling easy testing and demonstration of the system’s capabilities.

Technologies & Tools

Python, TensorFlow 2.15, Keras, Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, Seaborn, Gradio

Google Colab, Google Drive for data storage and model reproducibility

Key Outcomes

Successfully built a multi-label toxic comment classifier with strong performance on a real-world dataset.

Delivered a user-friendly deployment that demonstrates end-to-end model inference, from preprocessing to prediction.

Demonstrated proficiency in deep NLP pipelines, model tuning, and deployment of machine learning systems.

