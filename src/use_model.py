from pyspark.ml.pipeline import PipelineModel, Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, FloatType, DoubleType

#Variables
app_name = "Predict Titanic Survivor"
model_path = "./model/pipelinModel"

#Init SparkSession
spark = SparkSession \
    .builder \
    .appName(app_name) \
    .getOrCreate()

#Load the Model
classificationModel = PipelineModel.load(model_path)

# Read all the csv files written atomically in a directory
userSchema = StructType()\
    .add("Pclass",  FloatType())\
    .add("Age",  DoubleType())\
    .add("Sex",  StringType()) \
    .add("Fare", DoubleType()) \
    .add("Embarked", StringType())

csvDF = spark \
    .readStream \
    .option("sep", ";") \
    .schema(userSchema) \
    .csv("./data/input")  # Equivalent to format("csv").load("/path/to/directory")

predictions = classificationModel.transform(csvDF)

#Write stream to output file with a window of 10s
predictions.writeStream.format("json")\
    .trigger(processingTime="10 seconds")\
    .option("checkpointLocation", "checkpoint/")\
    .option("path", "./data/output") \
    .start() \
    .awaitTermination()
