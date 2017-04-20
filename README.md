# machinelearning tutorial in Pyspark ML library

### Read the Dataset
  - Description: the dataset including the target variable(default) and features
  - Rows: 10000
  - Columns(type): Default(bool) / Student(bool) / Balance(double) / Income(double)
  - 문제 타입 => Binary Classification
  - Target var: Default (채무불이행 여부, Skewed)
  - Features: Student, Balance, Income

### CSV & Caching
df = sqlContext.read.format('com.databricks.spark.csv')\
					.options(header='true', inferSchema='true')\
					.load('/Users/woowahan/Documents/Python/Pyspark_MLlib/data/Default.csv')\
					.drop("_c0")\
					.cache()
### 스키마 확인
df.printSchema()

### Descriptive Analysis
df.describe().show()
df.stat.crosstab("default", "student").show()
