# 📦 RetailMind
### Machine Learning-Based Demand Forecasting and Inventory Intelligence System

RetailMind is a machine-learning-based retail decision-support system designed to help businesses understand product demand, identify inventory risks, discover purchasing patterns, and make informed inventory management decisions.

The system combines demand forecasting, inventory intelligence, product clustering, and association rule mining through an interactive Streamlit dashboard and FastAPI backend.

---

## 🚀 Project Overview

Retail businesses face several challenges:

- Unpredictable product demand
- Stock shortages and overstocking
- Inefficient inventory planning
- Limited understanding of product purchasing patterns
- Difficulty making data-driven replenishment decisions

RetailMind addresses these challenges through four major analytical modules:

1. **Demand Forecasting:** Estimates product demand using machine learning.
2. **Inventory Intelligence:** Identifies stock risks and generates reorder recommendations.
3. **Product Clustering:** Groups store-product combinations based on their sales and inventory characteristics.
4. **Association Rule Mining:** Discovers products frequently purchased together.

RetailMind is designed as a decision-support system. It provides recommendations to assist human decision-making rather than automatically controlling inventory.

---

## 🏗️ System Architecture

```text
                 Retail Datasets
                       |
                       ▼
              Data Preprocessing
                       |
                       ▼
          Exploratory Data Analysis
                       |
                       ▼
              Feature Engineering
                       |
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
     Demand        Product       Association
   Forecasting    Clustering     Rule Mining
          |            |             |
          ▼            ▼             ▼
     Inventory     Cluster       Purchasing
   Intelligence    Analysis       Patterns
          |            |             |
          └────────────┼─────────────┘
                       |
                       ▼
               Processed Outputs
                       |
              ┌────────┴────────┐
              ▼                 ▼
          FastAPI           Streamlit
           Backend          Dashboard
              |                 |
              └────────┬────────┘
                       ▼
              RetailMind System
```

---

## 📊 Dataset Information

RetailMind uses two datasets for different analytical tasks.

### 1. Retail Store Inventory Dataset

Used for demand forecasting, inventory intelligence, and product clustering.

**Dataset characteristics:**

| Property | Details |
|---|---|
| Records | 73,100 |
| Stores | 5 |
| Products | 20 |
| Categories | 5 |
| Time period | 2022–2024 |
| Columns | 15 |
| Missing values | None initially |

Important features include:

- Date
- Store ID
- Product ID
- Category
- Region
- Inventory Level
- Units Sold
- Units Ordered
- Price
- Discount
- Weather Condition
- Holiday/Promotion
- Competitor Pricing
- Seasonality

The dataset is used as demonstration data. Its characteristics suggest synthetic or simulated retail data, so the results should not be interpreted as validation on real-world retail operations.

### 2. Online Retail Dataset

Used specifically for Association Rule Mining.

**Dataset source:** [UCI Machine Learning Repository – Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

Dataset characteristics:

- Original records: 541,909
- Cleaned records: 530,104
- Unique invoices: 19,960
- Unique products: 4,015

The transaction-level structure allows RetailMind to identify products purchased together within the same invoice.

---

## 🔍 Exploratory Data Analysis

The data analysis process includes:

- Missing value analysis
- Duplicate detection
- Numerical feature analysis
- Sales distribution analysis
- Category-wise demand analysis
- Store and product analysis
- Correlation analysis
- Time-based analysis
- Data quality assessment

The analysis helps identify data inconsistencies and understand the characteristics of the datasets before model development.

---

## ⚙️ Feature Engineering

Time-based and historical sales features were created for demand forecasting.

### Calendar Features

- Day
- Month
- Year
- Day of Week
- Weekend Indicator

### Lag Features

- Lag 1
- Lag 7
- Lag 14
- Lag 30

### Rolling Statistics

- Rolling Mean (7 Days)
- Rolling Mean (30 Days)

Categorical features were also converted into numerical representations using one-hot encoding.

The pre-existing Demand Forecast column was excluded from model inputs because of its unusually high correlation with actual sales and uncertain data provenance.

---

## 🤖 1. Demand Forecasting

RetailMind uses machine learning regression models to estimate product demand.

### Models Implemented

**Random Forest Regressor**

- Ensemble learning algorithm
- Uses multiple decision trees
- Captures nonlinear relationships
- Provides feature importance analysis

**XGBoost Regressor**

- Gradient boosting algorithm
- Builds trees sequentially
- Captures complex feature interactions
- Provides a comparison against Random Forest

### Model Evaluation

The models were evaluated using a chronological training and testing split.

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| MAE | 68.93 | 69.00 |
| RMSE | 88.17 | 88.32 |
| R² Score | 0.3333 | 0.3311 |

### Baseline Comparison

A previous-day sales baseline produced:

| Metric | Baseline | Random Forest |
|---|---:|---:|
| MAE | 119.58 | 68.93 |
| RMSE | 153.18 | 88.17 |

Random Forest achieved approximately 42.36% lower MAE than the baseline.

Random Forest performed marginally better than XGBoost on this dataset.

**Note:** These are experimental results on the selected demonstration dataset. Feature availability and data quality limitations mean that future operational forecasting performance has not yet been established.

---

## 📦 2. Inventory Intelligence

The inventory module analyzes stock availability and generates replenishment recommendations.

### Key Calculations

**Average Daily Demand**

Calculated using the previous seven days of sales.

**Stock Coverage**

```text
Stock Coverage = Current Inventory / Average Daily Demand
```

**Safety Stock**

```text
Safety Stock = Service Factor × Demand Standard Deviation × √Lead Time
```

**Reorder Point**

```text
Reorder Point = Average Daily Demand × Lead Time + Safety Stock
```

**Target Stock**

```text
Target Stock = Average Daily Demand × Target Days + Safety Stock
```

**Reorder Quantity**

```text
Reorder Quantity = Target Stock - Current Inventory
```

### Inventory Classification

| Status | Condition |
|---|---|
| Reorder Required | Stock coverage below 2 days |
| Healthy | Stock coverage between 2 and 7 days |
| Overstock | Stock coverage above 7 days |

The current demonstration configuration uses:

- Lead time: 2 days
- Target stock period: 7 days
- Service factor: 1.65

### Results

| Inventory Status | Store-Product Combinations |
|---|---:|
| Reorder Required | 53 |
| Healthy | 47 |
| Overstock | 0 |
| **Total** | **100** |

The system generates reorder quantities to assist inventory replenishment planning.

These thresholds and parameters are configurable demonstration assumptions rather than actual supplier policies.

---

## 🧩 3. Product Clustering

RetailMind uses unsupervised learning to identify groups of similar store-product combinations.

### Features Used

- Average Sales
- Average Inventory
- Average Price
- Average Discount
- Average Demand
- Sales Variability

The features are standardized using StandardScaler.

### Algorithms Implemented

**Hierarchical Clustering**

- Groups similar store-product combinations.
- Uses Ward linkage.
- Produces three clusters.

**DBSCAN**

- Density-based clustering algorithm.
- Identifies dense groups and potential outliers.
- Does not require a predefined number of clusters.

### Results

Hierarchical Clustering:

| Cluster | Combinations |
|---|---:|
| Cluster 0 | 43 |
| Cluster 1 | 37 |
| Cluster 2 | 20 |

Hierarchical Clustering Silhouette Score: **0.1661**

DBSCAN:

| Classification | Combinations |
|---|---:|
| Main cluster | 82 |
| Noise / Outliers | 18 |

DBSCAN identified only one cluster after excluding noise, so its silhouette score could not be calculated.

These results are exploratory and indicate relatively weak separation between the hierarchical clusters.

---

## 🛒 4. Association Rule Mining

RetailMind uses the Apriori algorithm to discover relationships between products purchased together.

Unlike the daily inventory dataset, transaction-level data is required for this module. Therefore, the UCI Online Retail dataset is used.

### Data Processing

1. Removed missing product descriptions.
2. Removed cancelled invoices.
3. Removed non-positive quantities.
4. Removed non-positive prices.
5. Grouped products by invoice number.
6. Converted invoices into binary transaction baskets.

### Apriori Configuration

| Parameter | Value |
|---|---:|
| Minimum Support | 0.01 |
| Maximum Itemset Length | 3 |
| Minimum Confidence | 0.30 |

### Association Rule Metrics

**Support:** Measures how frequently a product combination appears in all transactions.

**Confidence:** Measures how often the consequent products appear when the antecedent products are present.

**Lift:** Measures the strength of an association compared with independent occurrence.

### Results

| Metric | Result |
|---|---:|
| Frequent Itemsets | 1,883 |
| Generated Association Rules | 1,620 |
| Strong Association Rules | 771 |

The final rules were filtered using:

- Support ≥ 0.01
- Confidence ≥ 0.50
- Lift > 1

These relationships can help retailers explore cross-selling opportunities and product placement strategies.

---

## 📊 Streamlit Dashboard

RetailMind provides an interactive dashboard for viewing the outputs of its analytical modules.

### Business KPIs

- Total Store-Product Combinations
- Total Products
- Total Stores
- Reorder Required Count
- Inventory Risk Percentage

### Inventory Intelligence

- Inventory recommendation table
- Store-level filtering
- Inventory status breakdown
- Average daily demand by product
- Reorder recommendations

### Demand Forecasting

- Model-predicted demand by product
- MAE
- RMSE
- R² Score

### Product Clustering

- Hierarchical Clustering results
- DBSCAN results
- Cluster distribution
- Store-product cluster assignments

### Association Rule Mining

- Number of strong association rules
- Product antecedents
- Product consequents
- Support
- Confidence
- Lift

The dashboard currently displays saved analytical outputs generated through the notebooks.

---

## 🌐 FastAPI Backend

RetailMind includes a REST API built using FastAPI.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | API health check |
| POST | `/forecast` | Retrieve demand and inventory information |
| GET | `/inventory` | Retrieve inventory records |
| GET | `/association-rules` | Retrieve product association rules |

### Forecast Request Example

```json
{
  "store": "S003",
  "item": "P0010"
}
```

The API returns information such as:

- Predicted demand estimate
- Inventory level
- Average daily demand
- Stock coverage
- Reorder point
- Reorder quantity
- Inventory status

The API retrieves precomputed results from the project's reports directory. It does not currently perform live model inference.

### Interactive API Documentation

FastAPI provides Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Streamlit |
| Machine Learning | Scikit-learn, XGBoost |
| Clustering | Hierarchical Clustering, DBSCAN |
| Association Mining | Apriori, mlxtend |
| API Development | FastAPI, Uvicorn |
| Data Storage | CSV, Excel |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

---

## 📂 Project Structure

```text
RetailMind/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── assets/
│
├── data/
│   ├── retail_store_inventory.csv
│   └── Online Retail.xlsx
│
├── models/
│   ├── random_forest_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   ├── 01_retail_data_quality_assessment.ipynb
│   ├── 01_data_quality_assessment.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_inventory_optimization.ipynb
│   ├── 06_product_clustering.ipynb
│   └── 07_association_rule_mining.ipynb
│
├── reports/
│   ├── inventory_recommendations.csv
│   ├── demand_predictions.csv
│   ├── product_clustering_results.csv
│   ├── cluster_summary.csv
│   ├── association_rules.csv
│   └── frequent_itemsets.csv
│
├── streamlit_app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation and Execution

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project

```bash
cd Demand-Forecasting-Inventory-Optimization-master
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
pip install openpyxl mlxtend
```

### 5. Run the Streamlit Dashboard

```bash
python -m streamlit run streamlit_app/app.py
```

### 6. Run the FastAPI Backend

Open another terminal in the project folder:

```bash
python -m uvicorn api.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

**Important:** The reports and model files must be generated before running the dashboard and API. The required datasets must also be placed in the `data` folder.

---

## 🎯 Key Achievements

- Developed a multi-module retail decision-support system.
- Processed 73,100 retail inventory records.
- Implemented Random Forest and XGBoost demand forecasting.
- Compared machine learning models against a baseline.
- Developed inventory risk classification and reorder recommendations.
- Implemented Hierarchical Clustering and DBSCAN.
- Generated 1,883 frequent itemsets using Apriori.
- Identified 771 strong association rules.
- Built an interactive Streamlit dashboard.
- Integrated analytical outputs through FastAPI.
- Tested the API endpoints using Swagger UI.

---

## ⚠️ Limitations and Future Improvements

Current limitations:

- The inventory dataset is demonstration data and may not represent real retail operations.
- Forecasting performance is experimental and requires validation on real-world data.
- Some inventory feature timings need clarification before operational forecasting use.
- Inventory policies use configurable assumptions.
- Clustering separation is relatively weak on the current dataset.
- Association Rule Mining uses a separate transaction-level dataset.
- Dashboard and API currently consume precomputed outputs rather than performing real-time model inference.

Potential future improvements:

- Validate models using real retail datasets.
- Improve forecasting through hyperparameter optimization and additional validation.
- Integrate real-time model inference.
- Add more inventory optimization strategies.
- Improve clustering quality and interpretation.
- Expand dashboard filtering and reporting.
- Integrate additional retail transaction datasets.

---

## 👨‍💻 Project Information

**Project Name:** RetailMind

**Project Type:** Machine Learning-Based Retail Decision Support System

**Domain:** Machine Learning, Retail Analytics, Demand Forecasting and Inventory Intelligence

**Developed as:** Academic Project

---

