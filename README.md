# Credit Card Fraud Detection

This repository contains a comprehensive analysis and implementation of various machine learning models for detecting fraudulent credit card transactions. The project begins with an in-depth Exploratory Data Analysis (EDA) to understand the dataset's characteristics, followed by the training and evaluation of several classification models. The primary challenge addressed is the highly imbalanced nature of the dataset.

## Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23project-overview)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Workflow](https://www.google.com/search?q=%23workflow)
  - [Getting Started](https://www.google.com/search?q=%23getting-started)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation](https://www.google.com/search?q=%23installation)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [Models Implemented](https://www.google.com/search?q=%23models-implemented)
  - [Results](https://www.google.com/search?q=%23results)
  - [Technologies Used](https://www.google.com/search?q=%23technologies-used)

## Project Overview

The goal of this project is to build a robust model that can accurately identify fraudulent credit card transactions. Given the sensitive nature of fraud detection, the focus is on achieving high recall for the fraudulent class while maintaining reasonable precision. This script explores the data, visualizes patterns, and compares the performance of different classification algorithms.

## Dataset

The project uses the "Credit Card Fraud Detection" dataset from Kaggle.

  * **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  * **Description:** The dataset contains transactions made by European cardholders in September 2013. It presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.
  * **Features:**
      * It contains only numerical input variables which are the result of a PCA transformation. Features `V1, V2, ... V28` are the principal components obtained with PCA.
      * The only features that have not been transformed with PCA are 'Time' and 'Amount'.
      * The feature 'Class' is the response variable, and it takes value 1 in case of fraud and 0 otherwise.

## Workflow

1.  **Data Loading and Initial Inspection:** The dataset is loaded, and initial checks for shape, data types, and missing values are performed.
2.  **Exploratory Data Analysis (EDA):**
      * **Class Imbalance:** Visualized the severe class imbalance using a bar chart.
      * **Transaction Time Analysis:** Plotted the distribution of transaction times for both fraudulent and non-fraudulent transactions to identify temporal patterns. A new `Hour` feature was engineered for deeper analysis.
      * **Transaction Amount Analysis:** Analyzed and compared the statistical properties and distributions of transaction amounts for both classes.
      * **Correlation Analysis:** A heatmap was generated to visualize the correlation between different features.
      * **Feature Distribution:** Kernel Density Estimate (KDE) plots were created for each PCA component (`V1`-`V28`) to observe which features have different distributions for fraudulent vs. non-fraudulent transactions.
3.  **Data Preprocessing:** The data was split into training, validation, and test sets.
4.  **Model Training and Evaluation:** Several machine learning models were trained and evaluated on their ability to predict fraud.
5.  **Cross-Validation:** A K-Fold cross-validation strategy was implemented with LightGBM to ensure the model's performance is robust and generalizable.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You will need Python 3.x installed, along with the following libraries:

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
lightgbm
```

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a `requirements.txt` file** with the prerequisites listed above and install them:

    ```sh
    pip install -r requirements.txt
    ```

3.  **Download the dataset:**

      * Download `creditcard.csv` from the [Kaggle link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
      * Place the `creditcard.csv` file in the root directory of the project.

## Usage

The project is structured as a Python script/Jupyter Notebook. To run the analysis and train the models, simply execute the script or run the cells of the notebook sequentially in an environment like JupyterLab, VS Code, or Google Colab.

```sh
# If using a notebook, start the Jupyter server
jupyter notebook
```

Then, open the notebook file and run the cells.

## Models Implemented

The following classification models were trained and evaluated:

1.  **Random Forest Classifier:** An ensemble method based on decision trees.
2.  **AdaBoost Classifier:** A boosting algorithm that combines weak learners sequentially.
3.  **XGBoost:** A highly optimized gradient boosting library known for its performance and speed.
4.  **LightGBM:** A fast, distributed, high-performance gradient boosting framework.

## Results

The models were primarily evaluated using the **Area Under the ROC Curve (AUC)** score, which is suitable for imbalanced classification problems.

  * The gradient boosting models (**XGBoost** and **LightGBM**) demonstrated superior performance, achieving high AUC scores.
  * Feature importance analysis from these models revealed that certain PCA components like `V17`, `V14`, `V12`, and `V10` were highly predictive of fraud.
  * The final evaluation using a LightGBM model with K-Fold cross-validation provided a robust AUC score, confirming the model's effectiveness. The final classification report shows detailed precision, recall, and F1-scores for both classes.

## Technologies Used

  * **Programming Language:** Python 3
  * **Libraries:**
      * **Data Manipulation:** Pandas, NumPy
      * **Data Visualization:** Matplotlib, Seaborn, Plotly
      * **Machine Learning:** Scikit-learn, XGBoost, LightGBM
