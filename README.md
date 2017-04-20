# Machine Learning Tutorial in Pyspark ML Library

### the Dataset
  - Description: the dataset including the target variable(default) and features
  - Rows: 10000
  - Columns(type): Default(bool) / Student(bool) / Balance(double) / Income(double)
  - Issue => Binary Classification
  - Target var: Default (Skewed)
  - Features: Student, Balance, Income

### Pipeline
- API 문서: http://takwatanabe.me/pyspark/index.html
- 개요/설명 (출처: https://spark.apache.org/docs/latest/ml-guide.html)
- Featurization, Pipelines, Persistence, Utilities 등 서비스/기능 지원
- DataFrame-based API is primary API 
- RDD 기반 spark.mllib package는 당분간 유지 모드.. Spark 3.0부터 삭제될 예정)
- 보다 유저-프렌들리한 DataFrames (with Pipeline) 중심으로 발전 예정

### Pipeline 구성요소: Transformer, Estimator, Parameter
- Transformer: Scale, 선형변환, 벡터화, Prediction 등 기존 변수의 형태를 변환
- Estimator: 모델 알고리즘을 통해 데이터 learning 하는 과정
- Parameter: 정규화 등 각 모델에 맞는 특정 파라메터 설정
- Pipeline Stage 구성 예시
- StringIndexer: 문자를 숫자로 변환
- OneHotEncoder: 카테고리 변수를 더미 코딩
- VectorAssembler: 백터화
- StandardScaler: 표준점수 변환
- LinearRegression: 선형회귀모델


