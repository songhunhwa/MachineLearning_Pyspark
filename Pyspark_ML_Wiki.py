####################################################################################################
# 1. Data Preparation
####################################################################################################

# Let's read the csv file
# You may have to use the databricks's library to read the dataset
df = sqlContext.read.format('com.databricks.spark.csv')\
					.options(header='true', inferSchema='true')\
					.load('/Users/woowahan/Documents/Python/Pyspark_MLlib/data/Default.csv')\
					.drop("_c0")\
					.cache()

# The type of the dataset will be Spark dataframe if the library was used
# If not Spark dataframe, do change the type to Spark dataframe so the codes will run appropriately
# It should yield "<class 'pyspark.sql.dataframe.DataFrame'>"
type(df) 

# Take a look at first 5 rows and schema info.
df.show(5, False)
df.printSchema()

# You can check basic statistics by using pandas after importing it
import pandas as pd

df.toPandas().info()
df.toPandas().groupby('default').describe()
df.stat.crosstab("default", "student").show()
df.stat.corr('income', 'balance')

####################################################################################################
# 2. Preprocessing & Pipeline
####################################################################################################

# Let's do a little bit of preprocessing so that the model will predict well
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# There are categorical variables recorded as string so it has to be changed numerical variables
strIdx = StringIndexer(inputCol = "student", outputCol = "studentIdx")

# After converting, you can apply onehot encording to deal with categorical variables
encode = OneHotEncoder(inputCol = "studentIdx", outputCol = "studentclassVec")

# Let's apply the same procedure to the label(target) variable
# No need to apply onehot encoding to label (only string indexing is required)
label_StrIdx = StringIndexer(inputCol = "default", outputCol = "label")

# Build the first stages for the pipeline 
stages = [strIdx, encode, label_StrIdx]

# For numerical variables, let's do transform those to standard scaled variables
from pyspark.sql.functions import col, stddev_samp

numCols = ['income', 'balance']
for c in numCols:
	df = df.withColumn(c + "Scaled", col(c)/df.agg(stddev_samp(c)).first()[0])

# Finally, you can define the inputs for mordel
# In this case, the vector of the categorical variables and the scaled numerical variables were assinged
inputs = ["studentclassVec", "incomeScaled", "balanceScaled"]

# As all input features need to be vectorized, VectorAssembler function has to be used
assembler = VectorAssembler(inputCols = inputs, outputCol = "features")

# Add the assembler to the previous stage
stages += [assembler]

# Put stages to build the Pipeline
# - The stage consists of string indexer, onehot encoder, scaler, and vector assembler   
pipeline = Pipeline(stages = stages)

# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
pipelineModel = pipeline.fit(df)
dataset = pipelineModel.transform(df)

# Select the features in which you are interested.
originalCols = df.columns
selectedScaledCols = ["label", "features"] + originalCols

# The dataset for machine learning is finally ready
dataset = dataset.select(selectedScaledCols) 

####################################################################################################
# 3. Learning the Dataset with Algorithms
####################################################################################################

# For cross validation, let's split the dataset with the ratio of 7 & 3
(train, test) = dataset.randomSplit([0.7, 0.3], seed = 14)

# To check if randomly distributed well
train.groupby("label").count().show()
test.groupby("label").count().show()

# Creating initial Logistic Regression model which is known for binary classification
# You can use a lot of models, provided by Pyspark ML libray other than logistic regression
# For instance, Random Forest, SVM, Naive Bayes, and NN are available
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol = "label", featuresCol = "features", maxIter=10)

# Learn the training data by using logistic regression algorithm
# fit() method is used to learn the dataset
lrModel = lr.fit(train)

# Make predictions on the unseen data by using the transform() method.
predictions = lrModel.transform(test)

# Let's check the first 20 results
predictions.show()

####################################################################################################
# 4. Evaluation of the Model
####################################################################################################

# Now that you have done with the prediciton process, let's evaluate the model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# First, you need to build an evaluator 
# "rawPredictionCol" can be either rawPrediction or probability (Either way will yield the same result)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

# AUC and AUC(PR) are the metrics, provided by the library
print evaluator.getMetricName(), "The AUC of the Model is {}".format(evaluator.evaluate(predictions))
print "The AUC under PR curve is {}".format(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))

# If you want to check out the deufalt matrix
evaluator.getMetricName() 

# Some other matrix can be found as follows
print 'Model Intercept: ', lrModel.interceptVector
print 'Model coefficientMatrix: ', lrModel.coefficientMatrix

####################################################################################################
# 5. Param Setting
####################################################################################################

# To extract the best params from the model, let's use ParamGridBuilder along with CV
# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.05, 0.1, 0.5, 2.0]) # inverse of regularization strength; must be a positive float. smaller values specify stronger regularization
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.5, 1.0]) # a hybrid of L1L1 and L2L2 regularization
             .addGrid(lr.maxIter, [1, 5, 10, 20]) # Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge
             .build())

# By setting αα properly, elastic net contains both L1L1 and L2L2 regularization as special cases. 
# For example, if a linear regression model is trained with the elastic net parameter αα set to 11, 
# it is equivalent to a Lasso model. On the other hand, if αα is set to 00, the trained model reduces to a ridge regression model. 
#We implement Pipelines API for both linear regression and logistic regression with elastic net regularization.

print lr.explainParams()

# Create CV variable to run K-fold CV
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5) # you can set the number of folds(K)

# Run the cross validations with fit() & transform() methods
cvModel = cv.fit(train)
predictions = cvModel.transform(test)

# After fitting, check out the metrics if the model performed well compared to the previous one
evaluator.evaluate(predictions)

print 'Model Intercept: ', cvModel.bestModel.intercept
print 'Model coefficientMatrix: ', cvModel.bestModel.coefficientMatrix

####################################################################################################
# 6. Saving the Pipeline & Model for Re-use
####################################################################################################

# Save the pipeline
pipelinePath = './default_pipeline'
pipeline.write().overwrite().save(pipelinePath)

loadedPipeline = Pipeline.load(pipelinePath)
loadedPipelineModel = loadedPipeline.fit(df)

# Save the model
from pyspark.ml import PipelineModel

modelPath = './default_pipeline'
lrModel.write().overwrite().save(modelPath)

loadedPipelineModel = PipelineModel.load(modelPath)
test_reloadModel = loadedPipelineModel.transform(test)










