from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
import wget

#Variables
url_train_file = 'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv'
file_path_train_data = './data/train/train.csv'
directory_path_pipeline = './model/pipelinModel'
app_name = 'Train Titanic Model'

#Download test file in ./data directory
wget.download(url_train_file, file_path_train_data)

#Init Spark Session
spark = SparkSession \
    .builder \
    .appName(app_name) \
    .getOrCreate()

#Read train File
training_data = (spark.read.format("csv").option('header', 'true').load("./data/train/train.csv"))

#Select the needed messages
training_data = training_data.select(
                         training_data['Survived'].cast('float'),
                         training_data['Pclass'].cast('float'),
                         training_data['Sex'],
                         training_data['Age'].cast('float'),
                         training_data['Fare'].cast('float'),
                         training_data['Embarked']
                        )

#Replace None value
training_data = training_data.replace('?', None).dropna(how='any')

# Now, the Spark ML library only works with numeric data. But we still
# want to use the Sex and the Embarked column. For that, we will need to
# encode them with StringIndexer
# Configure an ML pipeline,
# which consists of three stages: stringIndexerSex, stringIndexerEmbarked, assembler and Logistic Regression
stringIndexerSex = StringIndexer(inputCol='Sex',outputCol='Gender',handleInvalid='skip')
stringIndexerEmbarked = StringIndexer(inputCol='Embarked',outputCol='Boarded',handleInvalid='skip')
required_features = [
                    'Pclass',
                    'Age',
                    'Gender',
                    'Fare',
                    'Boarded'
                   ]
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
lr = LogisticRegression(labelCol="Survived", featuresCol="features")
pipeline = Pipeline(stages=[stringIndexerSex, stringIndexerEmbarked, assembler, lr])

# Fit the pipeline
model = pipeline.fit(training_data)

#Save the model
model.save(directory_path_pipeline)