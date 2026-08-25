# AI-Powered Customer Complaint Triage & Priority Prediction

## Problem Statement
Customer support teams receive large numbers of free-text complaints every day. Manually reading, categorizing, and prioritizing every complaint is slow and can cause urgent issues to be handled late.

This project builds a simple NLP and machine-learning system that automatically:
1. Predicts the complaint category.
2. Predicts complaint priority as **Low, Medium, or High**.
3. Displays the prediction through a small Streamlit web app.

The goal is not to replace customer-support agents, but to provide a first-pass triage system that helps route complaints faster.

## Approach

### 1. Data
A small, realistic customer-complaint dataset is included in `data/complaints.csv`. Each record contains complaint text, category, and priority.

### 2. Text preprocessing
- Convert text to lowercase.
- Remove unnecessary punctuation.
- Keep useful words for classification.

### 3. Feature engineering
Complaint text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### 4. Machine learning
Two lightweight supervised classification models are trained:
- **Complaint Category:** Logistic Regression
- **Priority:** Logistic Regression

Using two models makes the system easy to understand and extend while keeping inference fast.

### 5. Evaluation
The training script reports:
- Accuracy
- Classification report
- Confusion matrix

A stratified train/test split is used where possible.

### 6. Application
A Streamlit interface accepts a new complaint and returns:
- Predicted category
- Predicted priority
- Model confidence
- Suggested routing team

## Key Insights
- Free-text complaints contain useful signals for automated triage; words related to refunds, delivery, login/access, billing, and technical failures can help distinguish complaint types.
- Priority can be estimated from urgency signals such as words describing blocked access, failed transactions, missing orders, or repeated unresolved problems.
- TF-IDF + Logistic Regression is a strong baseline for small text-classification projects because it is fast, interpretable, and does not require a large language model.
- In a real production system, the model should be retrained with historical support tickets and human-reviewed labels before being used for operational decisions.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK-style text preprocessing using Python/regex
- Streamlit
- Matplotlib / Seaborn for evaluation visualizations
- Git & GitHub

## Project Structure

```text
ai-powered-customer-complaint-triag/
├── data/
│   └── complaints.csv
├── models/
│   └── .gitkeep
├── src/
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/JaiperumalJayaraman/ai-powered-customer-complaint-triag.git
cd ai-powered-customer-complaint-triag
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the models
```bash
python src/train.py
```

This creates the trained models inside `models/` and prints evaluation metrics.

### 5. Run the Streamlit application
```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## Example

**Input:**
> I was charged twice for the same order and need the extra payment refunded immediately.

**Possible output:**
- Category: Billing / Refund
- Priority: High
- Routing: Billing & Payments Team

The exact prediction depends on the model trained from the supplied dataset.

## Limitations
- The included dataset is intentionally small and is suitable for a portfolio demonstration, not production use.
- Predictions may be wrong for complaints that are very different from the training examples.
- Priority labels are subjective and should ultimately be validated by support policies and human agents.

## Future Improvements
- Train on thousands of anonymized historical complaints.
- Add multilingual complaint support.
- Add sentiment analysis.
- Add duplicate-complaint detection.
- Add human-in-the-loop feedback so agents can correct predictions.
- Deploy the application using Streamlit Community Cloud or another hosting platform.

## Resume Description
**AI-Powered Customer Complaint Triage & Priority Prediction** — Built an NLP-based complaint triage system using TF-IDF and Logistic Regression to automatically classify customer complaints by category and priority; developed a Streamlit interface for real-time predictions and model confidence, with an end-to-end training and evaluation pipeline.
