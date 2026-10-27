# AI-Based Fake News Detection System

## Project Overview
This project implements an AI-based Fake News Detection System using a **Bidirectional Long Short-Term Memory (BiLSTM)** deep learning model. It classifies English news text as Real or Fake through an interactive web application built with Streamlit.

The application also integrates **Local Interpretable Model-Agnostic Explanations (LIME)** to highlight words that influence the model's prediction.

## Features
- **News Classification:** Predicts whether submitted news text is Real or Fake.
- **Confidence Scores:** Displays the predicted probabilities for both classes.
- **Model Explainability:** Uses LIME to provide word-level explanations.
- **Interactive Interface:** Provides a Streamlit web application for testing news headlines and articles.
- **Example Inputs:** Includes sample news text for quick demonstrations.

## Technologies Used
- Python 3.10+
- TensorFlow
- NumPy
- Streamlit
- NLTK
- LIME
- scikit-learn

## Dataset
The BiLSTM model was trained using a combined dataset based on:
- ISOT Fake News Dataset
- LIAR Dataset
- Kaggle Fake News Dataset

The datasets were combined for model development. Refer to the original sources for dataset details and licensing conditions.

### Dataset Access
The combined datasets are stored separately from this GitHub repository because of their file sizes.

- **Google Drive — Datasets and Additional Models:** [Access project assets](https://drive.google.com/drive/folders/1fpM-aoLGREI4P0-urBdGFvIkNB4u6f70?usp=drive_link)

Available files include:

- `combined_fake_news.csv`
- `combined_fake_news_preprocessed.csv`

Please check the original dataset licenses and redistribution conditions before sharing or reusing the data.

## Model and Artifacts
### BiLSTM — Current Application Model
The Streamlit application currently uses:
- `models/bilstm_savedmodel/` — Trained BiLSTM model in TensorFlow SavedModel format.
- `models/tokenizer.pkl` — Tokenizer used to convert text into sequences.

These files must be available at the expected paths for the application to run.

### Additional Machine-Learning Models
The following trained models are stored separately on Google Drive for preservation and comparative experiments:
- `logistic_regression_balanced.pkl`
- `random_forest_balanced.pkl`
- `xgboost_balanced.pkl`

**Google Drive — Additional Models:** [Access project assets](https://drive.google.com/drive/folders/19hjcOWW85ue7Iyt8JsSU-vWdf8HT46oG?usp=drive_link)

These models are not loaded by the current Streamlit application.

### DistilBERT Model
The trained DistilBERT model is hosted separately on Hugging Face because of its file size.

- **Hugging Face — DistilBERT Model:** [View model repository](https://huggingface.co/Reywerd/fake-news-distilbert)

The repository contains the model weights and supporting configuration and tokenizer files. The DistilBERT model is retained for comparative experiments and is not currently loaded by the Streamlit application.

## Project Structure
```text
fake-news-detection/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   ├── bilstm_savedmodel/
│   └── tokenizer.pkl
└── notebooks/
    ├── Colab_BiLSTM.ipynb
    ├── Colab_Data_Collection.ipynb
    ├── Colab_DistilBERT.ipynb
    └── Local_Fake_News_Detection_Project.ipynb
```
*Note: This illustrates the recommended repository structure. Large datasets and model artifacts may be hosted separately.*

## Requirements
- Python 3.10 or a compatible version supported by the dependencies.
- pip
- The trained BiLSTM model and tokenizer files.

## Installation and Setup
1. Clone the repository:

   ```bash
   git clone https://github.com/JiaWei2918/Fake_News_Detection_System.git
   cd Fake_News_Detection_System
   ```

2. Create a virtual environment:

   ```bash
   python -m venv fake_news_env
   ```

3. Activate it on Windows:

   ```bash
   fake_news_env\Scripts\activate
   ```

4. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Ensure the trained model and tokenizer are available at the paths expected by `app.py`.
## How to Run
Start the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Usage
1. Enter an English news headline or article into the text area.
2. Click **Check News**.
3. Review the predicted classification and confidence scores.
4. Expand **Why did the model decide this?** to view the LIME explanation.

Explanation generation may take approximately 60–90 seconds, depending on the input text and system performance.

## Evaluation
The application displays a reported test accuracy of **90.03%**. This result should be interpreted in the context of the dataset, preprocessing pipeline, test split, and evaluation methodology used during model development.

## Limitations
- The model is designed for English-language news text.
- Predictions may be incorrect and should not be treated as definitive fact-checking results.
- LIME explanations are approximations of model behaviour, not proof that a news claim is true or false.
- Explanation generation can be computationally intensive for longer articles.
- The application is intended for educational and research purposes.

## Acknowledgements
This project uses TensorFlow, Streamlit, NumPy, NLTK, LIME, scikit-learn, and datasets from ISOT, LIAR, and Kaggle.
