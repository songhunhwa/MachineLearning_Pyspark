# read the csv with library
df = sqlContext.read.format('com.databricks.spark.csv')\
					.options(header='true', inferSchema='true')\
					.load('/Users/songhunhwa/Documents/Python/Pyspark_MLlib/data/Default.csv')\
					.drop("_c0")\
					.cache()

# schema
df.printSchema()

# basic stat
import pandas as pd

df.toPandas().info()
df.toPandas().groupby('default').describe()
df.stat.crosstab("default", "student").show()
df.toPandas().corr()
#df.stat.corr('income', 'balance')

## preprocessing
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# cate vals
strIdx = StringIndexer(inputCol = "student", outputCol = "studentIdx")
encode = OneHotEncoder(inputCol = "studentIdx", outputCol = "studentclassVec")
label_StrIdx = StringIndexer(inputCol = "default", outputCol = "label")
stages = [strIdx, encode, label_StrIdx]

# scaling 
from pyspark.sql.functions import col, stddev_samp

numCols = ['income', 'balance']
for c in numCols:
	df = df.withColumn(c + "Scaled", col(c)/df.agg(stddev_samp(c)).first()[0])

# vectorize
inputs = ["studentclassVec", "incomeScaled", "balanceScaled"]
assembler = VectorAssembler(inputCols = inputs, outputCol = "features")
stages += [assembler]

# Pipeline
pipeline = Pipeline(stages = stages)

# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
pipelineModel = pipeline.fit(df)
dataset = pipelineModel.transform(df)

# select features
originalCols = df.columns
selectedScaledCols = ["label", "features"] + originalCols

dataset = dataset.select(selectedScaledCols) 

# split
(train, test) = dataset.randomSplit([0.7, 0.3], seed = 14)

train.groupby("label").count().show()
test.groupby("label").count().show()

# Create initial LogisticRegression model
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol = "label", featuresCol = "features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(train)

# Make predictions on test data using the transform() method.
predictions = lrModel.transform(test)

predictions.show()
predictions.groupby("prediction").count().show()
predictions.groupby("label").count().show()

#selected = predictions.select("label", "student", "balance")

# evaluations
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# rawPredictionCol can be either rawPrediction or probability
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

#evaluator.evaluate(predictions)
print evaluator.getMetricName(), "The AUC of the Model is {}".format(evaluator.evaluate(predictions))
print "The AUC under PR curve is {}".format(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))

evaluator.getMetricName() #AUC is deufalt matrix

print 'Model Intercept: ', lrModel.interceptVector
print 'Model coefficientMatrix: ', lrModel.coefficientMatrix


# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0]) #inverse of regularization strength; must be a positive float. smaller values specify stronger regularization
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) #a hybrid of L1L1 and L2L2 regularization
             .addGrid(lr.maxIter, [1, 5, 10]) #U seful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge
             .build())

# By setting αα properly, elastic net contains both L1L1 and L2L2 regularization as special cases. 
# For example, if a linear regression model is trained with the elastic net parameter αα set to 11, 
# it is equivalent to a Lasso model. On the other hand, if αα is set to 00, the trained model reduces to a ridge regression model. 
#We implement Pipelines API for both linear regression and logistic regression with elastic net regularization.

print lr.explainParams()

cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(train)
predictions = cvModel.transform(test)
evaluator.evaluate(predictions)

print 'Model Intercept: ', cvModel.bestModel.intercept
print 'Model coefficientMatrix: ', cvModel.bestModel.coefficientMatrix

# save the pipeline
pipelinePath = './default_pipeline'
pipeline.write().overwrite().save(pipelinePath)

loadedPipeline = Pipeline.load(pipelinePath)
loadedPipelineModel = loadedPipeline.fit(df)

# save the model
from pyspark.ml import PipelineModel

modelPath = './default_pipeline'
lrModel.write().overwrite().save(modelPath)

loadedPipelineModel = PipelineModel.load(modelPath)
test_reloadModel = loadedPipelineModel.transform(test)












