# Machine Learning Tutorial in Pyspark ML Library

### Info.
  - This documnet includes the way of how to run machine learning with Pyspark ml libaray. 
  - It was based on PySpark version 2.1.0 (Python 2.7). 
  - Below Spark version 2, pyspark mllib was the main module for ML, but it entered a maintenance mode.
  - Instead, at spark 2 verion, pyspark ml module became a main module. 
  - Therefore, this doc was created based on pyspark.ml module.

### Dataset
  - Description: the dataset including the target variable(default) and features
  - Rows: 10000
  - Columns(type): Default(bool) / Student(bool) / Balance(double) / Income(double)
  - Issue => Binary Classification
  - Target var: Default (Skewed)
  - Features: Student, Balance, Income

### Pipeline
  - API docs: http://takwatanabe.me/pyspark/index.html
  - Overivew: (https://spark.apache.org/docs/latest/ml-guide.html
  - Featurization, Pipelines, Persistence, Utilities
  - DataFrame-based API is primary API 
  - RDD-based spark.mllib package will be depricated from Spark 3.0)
  - As DataFrames (with Pipeline) is more user-friendly, this data type will be more frequently used.
  
### Pipeline components: Transformer, Estimator, Parameter
  - Transformer: Scale, linear transformation, vectorize, Prediction
    - Estimator: learning the data via a modeling algorithm
    - Parameter: Regulization, the number of iterations
  - Pipeline Stage Examples
    - StringIndexer: Convert string to index
    - OneHotEncoder: Convert a categorical variable to dummies
    - VectorAssembler: vectorize
    - StandardScaler: transform the original values to Z-score
    - LinearRegression: the famous model for predicting real numbers

### Learn the dataset with algorithm
  - Reference: https://spark.apache.org/docs/latest/ml-classification-regression.html
  - Classification
    - Logistic regression
    - Decision tree classifier
    - Random forest classifier
    - Gradient-boosted tree classifier
    - Multilayer perceptron classifier
    - One-vs-Rest classifier (a.k.a. One-vs-All)
    - Naive Bayes
  - Regression
    - Linear regression
    - Generalized linear regression
    - Decision tree regression
    - Random forest regression
    - Gradient-boosted tree regression
    - Survival regression
    - Isotonic regression
  - Clustering
    - K-means
    - Latent Dirichlet allocation (LDA)
    - Bisecting k-means
    - Gaussian Mixture Model (GMM)

### Evaluation / Parameter Tuning
  - Model selection (a.k.a. hyperparameter tuning)
  - Cross-Validation
  - Train-Validation Split
  
### Save & Load the Pipeline and Model
