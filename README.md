# ESG News Classification with DistilBERT

## Project Scope
The project aims to develop a machine learning system that automatically analyzes news articles to detect **ESG-related incidents** and classify them into **Environmental (E), Social (S), or Governance (G)** categories.  

**ESG** stands for *Environmental, Social, and Governance*:
- **Environmental (E):** emissions, resource usage, climate impact.
- **Social (S):** labor practices, human rights, community impact.
- **Governance (G):** corporate ethics, transparency, management practices.

---

## Project Structure
- `ESG_Analysis_MultilabelModel.ipynb`: Main notebook for preprocessing, model training, and evaluation.
- `best_distilbert_esg/`: Directory containing the best trained DistilBERT model.
  - `config.json`
  - `model.safetensors`
  - `special_tokens_map.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `training_args.bin`
  - `vocab.txt`
- `Agent.py`: ESG Agent implementation code.
- `Load_Model.py`: Example script to load model and run predictions.
- `requirements.txt`: Dependencies list.
- `Automated ESG News Analysis for Businesses.pdf`: Project report (methodology, experiments, results).


## Data and file dependencies

The data is available in a .zip file on Google Drive https://drive.google.com/drive/folders/1WIuk00-wVpbg6zDlzpH559KX4xzgZQKz?usp=sharing under file data_n_dependencies, accompanied by the files articles_labeling.xlsx and keywords.xlsx, which are used in the analysis.

---

## Methodology
1. **Data Preprocessing**
   - Cleaned and preprocessed text from ESG-related news datasets.
   - Tokenization using HuggingFace’s DistilBERT tokenizer.

2. **Modeling**
   - Base model: `distilbert-base-uncased`.
   - Fine-tuned on multilabel classification (E, S, G).
   - Evaluation metrics: F1-score (macro/micro), Precision, Recall.

3. **Training**
   - Saved best-performing checkpoint in `best_distilbert_esg/`.

4. **Evaluation**
   - Achieved strong performance across ESG categories.
   - Final model exported for reproducibility.

---

## Usage

### 1. Clone Repository
```bash
git clone https://github.com/chrysakaragkouni-cmd/esg-distilbert.git
cd esg-distilbert
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Notebook
```bash
jupyter notebook ESG_Analysis_MultilabelModel.ipynb
```

### 4. Run Prediction Example Script
```bash
python load_model.py
```

## 5. Run Aget
```bash
python streamlit run Agent.py
```

---

## Model Weights
Due to GitHub’s file size limits, the fine-tuned model is included in a `.zip` file via Google Drive https://drive.google.com/drive/folders/1WIuk00-wVpbg6zDlzpH559KX4xzgZQKz?usp=sharing under name best_distilbert_esg.

---

## Requirements
Key dependencies are listed in `requirements.txt`:
- pandas
- numpy
- scikit
- learn
- matplotlib
- seaborn
- nltk
- spacy
- unidecode
- torch
- transformers
- sentence
- transformers
- xgboost
- shap
- lime
- tqdm
- datasets
- openpyxl
- streamlit
- requests
- beautifulsoup4
- python
- dotenv

---

## Authors
Project developed by **November** team as part of the Machine Learning and Content Analytics project.
