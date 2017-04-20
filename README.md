# machinelearning tutorial in Pyspark ML library

### Read the Dataset
  - Description: the dataset including the target variable(default) and features
  - Rows: 10000
  - Columns(type): Default(bool) / Student(bool) / Balance(double) / Income(double)
  - 문제 타입 => Binary Classification
  - Target var: Default (채무불이행 여부, Skewed)
  - Features: Student, Balance, Income

### CSV 불러오기 & Caching
df = sqlContext.read.format('com.databricks.spark.csv')\
					.options(header='true', inferSchema='true')\
					.load('/Users/woowahan/Documents/Python/Pyspark_MLlib/data/Default.csv')\
					.drop("_c0")\
					.cache()
 
### 스키마 확인
df.printSchema()
root
 |-- default: string (nullable = true)
 |-- student: string (nullable = true)
 |-- balance: double (nullable = true)
 |-- income: double (nullable = true)
 
#### Descriptive Analysis
df.describe().show()
+-------+-------+-------+-----------------+-----------------+
|summary|default|student|          balance|           income|
+-------+-------+-------+-----------------+-----------------+
|  count|  10000|  10000|            10000|            10000|
|   mean|   null|   null| 835.374885614945|33516.98187595726|
| stddev|   null|   null|483.7149852103292|13336.63956273191|
|    min|     No|     No|              0.0|      771.9677294|
|    max|    Yes|    Yes|      2654.322576|       73554.2335|
+-------+-------+-------+-----------------+-----------------+
 
df.stat.crosstab("default", "student").show()
+---------------+----+----+
|default_student|  No| Yes|
+---------------+----+----+
|             No|6850|2817|
|            Yes| 206| 127|
+---------------+----+----+
