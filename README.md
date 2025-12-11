# AutoAnalyst AI  
## Turning Raw Data Into Clear, Explainable Intelligence

AutoAnalyst AI is an end-to-end Streamlit application that transforms raw datasets into insights, models, and SQL — fast, accurate, and explainable.  
Upload a dataset and AutoAnalyst AI delivers:

 Cleaned + prepared data  
 Automated Exploratory Data Analysis (EDA)  
 Interactive visual insights  
 Guided AutoML (classification & regression)  
 Natural Language → SQL translation  
 Easy exporting of data, SQL, and models  

---

##  Features

###  Data Upload
- Supports **CSV** & **Excel** (preview & basic stats)  
- Instant dataset preview  
- Session-based data storage  
- Clear/reset dataset option

###  Simple Data Cleaning
- Drops rows containing missing values  
- Optional column dropping  
- Cleaning audit: rows removed, columns dropped, before/after shape  
- Cleaned output stored as `__cleaned_df__`

###  Visual EDA
- Summary statistics  
- Missing-value heatmap  
- Correlation matrix (with ID-column filtering)  
- Outlier detection  
- Category distributions  
- Human-like narrative summary highlighting key insights

###  Interactive Visualizations
- Histograms, boxplots, scatter plots, bar charts  
- Correlation and pairwise views  
- Dashboard-ready charts and exportable images

###  AutoML Pipeline
- Support for classification & regression workflows  
- Choose target variable and features  
- Train models (Logistic Regression, Random Forest, etc.)  
- Evaluate using Accuracy, F1, AUC, RMSE, and related metrics  
- Feature importance visualizations  
- Predict on new data and export trained models

###  Natural Language → SQL
- Translate English queries (single-table focus) into SQL  
- Support for WHERE, GROUP BY, ORDER BY, LIMIT, aggregations  
- Fuzzy column matching for tolerant parsing  
- Optional sandboxed execution using SQLite for validation  
- Download generated SQL

###  Export System
- Download cleaned datasets (CSV)  
- Download generated SQL (.sql)  
- Export trained models (pickle)

---

##  Tech Stack

- **Frontend / App:** Streamlit  
- **Data Processing:** pandas, numpy  
- **Visualization:** matplotlib, seaborn, plotly  
- **Machine Learning:** scikit-learn, shap  
- **SQL Sandbox:** sqlite3  
- **Deployment:** Local / Streamlit Cloud / Cloud providers

---

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/AutoAnalyst-AI.git
cd AutoAnalyst-AI
```
### 2. Create Environment
```bash
conda create -n autoanalyst python=3.10
conda activate autoanalyst
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Application
```bash
streamlit run app.py
```

---

##  Project Structure
```bash
AutoAnalyst-AI/
├── app.py                # Main Streamlit application (routing + UI)
├── data_upload.py        # Data ingestion (CSV/Excel) + session storage
├── data_cleaning.py      # Simple cleaning (drop NA + column removal)
├── eda.py                # Exploratory Data Analysis + narrative insights
├── automl.py             # ML training, metrics, SHAP explainability
├── nl2sql.py             # Natural Language → SQL translation + sandbox
├── about.py              # About page content
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

##  Example Workflow

1) Upload your dataset (CSV/Excel) in the Data Upload section.
2) Clean the data by dropping missing-value rows or removing unnecessary columns.
3) Use EDA to explore data distributions, correlations, missingness, and narrative insights.
4) Convert English instructions to SQL using NL → SQL and optionally execute them in a sandbox.
5) Build ML models in AutoML, evaluate metrics, and interpret results with SHAP.
6) Predict on new samples and export cleaned data, SQL, or trained models.

---

## License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with attribution.

---------

## Author

Atharva Chavhan
Gmail- atharvachavhan18@gmail.com

