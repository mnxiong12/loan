import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer

# Resolve base directory of this script
base_dir = os.path.abspath(os.path.dirname(__file__))

# Construct full input and output paths
input_path = f"file://{os.path.join(base_dir, 'loan_data.csv')}"
output_path = os.path.join(base_dir, "output.txt")

# Initialize Spark session
spark = SparkSession.builder.appName('Loan').getOrCreate()

# Load dataset
df = spark.read.csv('loan_data.csv', header=True, inferSchema=True)

# *** NULLS ***
# Review dataset schema to understand columns and their datatype
df.printSchema()

# Determine which columns contains null and the number of nulls
for column in df.columns:
    print(column + ":", df[df[column].isNull()].count())

# For columns with numerical variables, instead of dropping the nulls, their values are replaced with their respective mean.
# Since the number of missing values are small (22 nulls and 14 nulls), replacing them with the mean instead allows us to preserve the other data within those rows.
imputer = Imputer(
    inputCols=['LoanAmount', 'Loan_Amount_Term'],
    outputCols=['LoanAmount', 'Loan_Amount_Term']
    ).setStrategy('mean')

df_copy = imputer.fit(df).transform(df)

# Groupby aggregations are used to help visualize columns with null values to see count distributions.
df.groupBy('Credit_History').count().show()
df.groupBy('Credit_History').count().show()
df.groupBy('Dependents').count().show()
df.groupBy('Married').count().show()
df.groupBy('Gender').count().show()

# For Credit_History, there are 50 null values, therefore dropping all rows associated with these null values would be quite a large amount of data deletion.
# A count reveals that there are 475 1s and 89 0s, so I decided to replace all nulls with 1s as that is the most common variable.
# The same strategy is used for the other categorical columns of 'Self_Employed', 'Dependents', 'Married', 'Gender'
df_copy = df_copy.na.fill({
    "Credit_History": 1,
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Self_Employed": "No"
})

# *** OUTLIER ***
# Outlier was determined using IQR. Here, all numerical columns are processed through the for loops to remove outliers from the columns.
columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

non_outlier_condition = None

for column in columns:
    quantiles = df_copy.approxQuantile(column, [0.25, 0.75], 0.05)
    IQR = quantiles[1] - quantiles[0]
    lower_bound = quantiles[0] - 1.5 * IQR
    upper_bound = quantiles[1] + 1.5 * IQR

    condition = (col(column) >= lower_bound) & (col(column) <= upper_bound)
    
    if non_outlier_condition is None:
        non_outlier_condition = condition
    else:
        non_outlier_condition = non_outlier_condition & condition

# Keep only rows that are within IQR bounds
clean_df = df_copy.filter(non_outlier_condition)

# A count of the new clean_df with outliers removed revealed that there are only now 416 rows. This is a significant decrease from the original 614 rows.
# Because the numerical columns will later be discretized, I decided to not drop the outliers. Doing so would mean a dataset that is significantly smaller, 
# which can introduce bias or create overfitting.
clean_df.count()

# *** DISCRETIZE COLUMNS ***
# The first step was understanding the data within the numerical columns, which is done here through looking at their mean, std dev, min, and max.
df_copy.describe(columns).show()

# Due to a wider range of values within the ApplicantIncome, I decided to split ApplicantIncome into more categories for better analysis.
# Note that the max value is 81000, which is significantly higher than the mean of 5403, suggesting potential outliers. 
# Therefore, anything higher than 12000 is grouped into its own category as majority are most likely outliers.
df_copy = df_copy.withColumn("cat_ApplicantIncome", 
       when(col("ApplicantIncome") <= 3000, "Lower")
      .when((col("ApplicantIncome") > 3000) & (col("ApplicantIncome") <= 6000), "Lower-Middle")
      .when((col("ApplicantIncome") > 6000) & (col("ApplicantIncome") <= 9000), "Middle")
      .when((col("ApplicantIncome") > 9000) & (col("ApplicantIncome") <= 12000), "Higher-Middle")
      .otherwise("Higher"))

# Another way in which I better understand the distribution is by grouping.
df_copy.groupBy('CoapplicantIncome').count().orderBy('count', ascending=False).show()

# A large number of CoapplicantIncome is zero (273 out of 614 rows). Therefore, I decided to seperate zero as its own category.
# Since the rest are not as spread apart, I decided to seperate them into three other categories, which the category of 'Higher' catching all the outliers.
df_copy = df_copy.withColumn("cat_CoapplicantIncome", 
       when(col("CoapplicantIncome") == 0, "Zero")
      .when((col("CoapplicantIncome") > 0) & (col("CoapplicantIncome") <= 2000), "Lower")
      .when((col("CoapplicantIncome") > 2000) & (col("CoapplicantIncome") <= 4000), "Middle")
      .otherwise("Higher"))

# I decided to split loans into intervals of 50.
df_copy = df_copy.withColumn("cat_LoanAmount", 
       when(col("LoanAmount") <= 50, "0-50")
      .when((col("LoanAmount") > 50) & (col("LoanAmount") <= 100), "50-100")
      .when((col("LoanAmount") > 100) & (col("LoanAmount") <= 150), "100-150")
      .when((col("LoanAmount") > 150) & (col("LoanAmount") <= 200), "150-200")
      .when((col("LoanAmount") > 200) & (col("LoanAmount") <= 250), "200-250")
      .otherwise("250+"))

# A look at the count by group to see how well the bins are distributed within cat_LoanAmount
df_copy.groupBy("cat_LoanAmount").count().show()

# Loan_Amount_Term is separated into three different bins since majority are over 300 (541 rows).
df_copy = df_copy.withColumn("cat_Loan_Term",
    when(col('Loan_Amount_Term') <= 120, "Short Term")
    .when((col('Loan_Amount_Term') > 120) & (col('Loan_Amount_Term') <= 300), "Medium Term")
    .otherwise("Long Term"))

# Drop columns that are note needed.
df2 = df_copy.drop('Loan_ID', 'Gender', 'Dependents', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term')

# *** EXPLAINATION WHY CERTAIN COLUMNS ARE SELECTED ***
# The columns of Loan_ID was dropped since it is an unique identifier and does not have any predictive information.
# The columns of Gender and Dependents were dropped since they are as closely tied to to finances. Additionally, Self_Employment was dropped since there was a big difference between 'No' and 'Yes' in that column (532 vs 82).
# Gender also have 112 Female vs 502 Male, which can lead to bias in the prediction. 
# Married and Education can be good indicators of financial standing, hence I left them. Instead of the numerical categories, I decided to stick to the discretized versions of those categories.
# All those categories were kept since they are closely tied to financial standing and can provide insight into loan approval status.

# Encode categorical variables by converting categorical columns into numerical values
married_indexer = StringIndexer(inputCol="Married", outputCol="MarriedIndexed")
education_indexer = StringIndexer(inputCol="Education", outputCol="EducationIndexed")
credit_indexer = StringIndexer(inputCol="Credit_History", outputCol="CreditHistoryIndexed")
property_indexer = StringIndexer(inputCol="Property_Area", outputCol="PropertyAreaIndexed")
applicantincome_indexer = StringIndexer(inputCol="cat_ApplicantIncome", outputCol="ApplicantIncomeIndexed")
coapplicantincome_indexer = StringIndexer(inputCol="cat_CoapplicantIncome", outputCol="CoapplicantIncomeIndexed")
loanamount_indexer = StringIndexer(inputCol="cat_LoanAmount", outputCol="LoanAmountIndexed")
loanterm_indexer = StringIndexer(inputCol="cat_Loan_Term", outputCol="LoanTermIndexed")
status_indexer = StringIndexer(inputCol="Loan_Status", outputCol="StatusIndexed")

indexers = [
    married_indexer, 
    education_indexer, 
    credit_indexer, 
    property_indexer,
    applicantincome_indexer, 
    coapplicantincome_indexer, 
    loanamount_indexer,
    loanterm_indexer, 
    status_indexer]

# Assemble features
assembler = VectorAssembler(
    inputCols = ['MarriedIndexed', 'EducationIndexed', 'CreditHistoryIndexed', 'PropertyAreaIndexed', 'ApplicantIncomeIndexed', 'CoapplicantIncomeIndexed', 'LoanAmountIndexed', 'LoanTermIndexed'],
    outputCol = 'features')

# *** DECISION TREE CLASSIFICATION ***
# Define decision tree classifier model
dt = DecisionTreeClassifier(labelCol="StatusIndexed", featuresCol="features")

# Build dt_pipeline
dt_pipeline = Pipeline(stages=indexers + [assembler, dt])

# Split the data
train_data, test_data = df2.randomSplit([0.8, 0.2], seed=42)

# Train dt_model
dt_model = dt_pipeline.fit(train_data)

# Generate predictions on test data
dt_predictions = dt_model.transform(test_data)

# Evaluate accuracry
dt_evaluator = MulticlassClassificationEvaluator(labelCol="StatusIndexed", predictionCol="prediction", metricName="accuracy")
dt_accuracy = dt_evaluator.evaluate(dt_predictions)

# *** RANDOM FOREST CLASSIFICATION ***
# Define random forest classification
rf = RandomForestClassifier(labelCol="StatusIndexed", featuresCol="features", numTrees=100)

# Build rf_pipeline
rf_pipeline = Pipeline(stages=indexers + [assembler, rf])

# Train rf_model
rf_model = rf_pipeline.fit(train_data)

# Make predictions
rf_predictions = rf_model.transform(test_data)

# Evaluate accuracy
rf_evaluator = MulticlassClassificationEvaluator(labelCol="StatusIndexed", predictionCol="prediction", metricName="accuracy")
rf_accuracy = rf_evaluator.evaluate(rf_predictions)

# *** NAIVE BAYES CLASSIFICATION ***
# Define naive bayes classification
nb = NaiveBayes(labelCol="StatusIndexed", featuresCol="features")

# Build nb_pipeline
nb_pipeline = Pipeline(stages=indexers + [assembler, nb])

# Train nb_model
nb_model = nb_pipeline.fit(train_data)

# Make predictions
nb_predictions = nb_model.transform(test_data)

# Evaluate accuracy
nb_evaluator = MulticlassClassificationEvaluator(labelCol="StatusIndexed", predictionCol="prediction", metricName="accuracy")
nb_accuracy = nb_evaluator.evaluate(nb_predictions)

# *** SUMMARY ***
# Model accuracies were: Decision Tree of 0.763, Random Forest of 0.773, and Naive Bayes of 0.773. Both Random Forest and Naive Bayes Classification were the most accurate,
# although not by much. The overall low accuracy level may have been affected by many factors. One likely due to using all categorical variables with potentially skewed/uneven bins.
# Redefining the bins may lead to better accuracy. Additionally, it is also possible to including too many features have lead to weaker predictions as they may introduce noise while fitting the model.

# Write accuracies to output file
with open(output_path, "w") as f:
    f.write(
        f"Decision Tree Classification Accuracy: {dt_accuracy:.4f}\n"
        f"Random Forest Classification Accuracy: {rf_accuracy:.4f}\n"
        f"Naive Bayes Classification Accuracy: {nb_accuracy:.4f}\n"
    )

print(f"Results written to {output_path}")


# Stop Spark

spark.stop()
