import wget
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import when, isnull
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.classification import LogisticRegression

#Variables
url_train_file = 'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv'
file_path_train_data = './data/train.csv'
directory_path_model = './model/crossValidatorModel'
directory_path_pipeline = './model/pipelinModel'
app_name = 'Train Titanic Model'

#Download test file in ./data directory
#wget.download(url_train_file, file_path_train_data)

#Init Spark Session
spark = SparkSession \
    .builder \
    .appName(app_name) \
    .getOrCreate()

#Read train File
training_data = (spark.read.format("csv").option('header', 'true').load("./data/train/train.csv"))

#Select the needed messages
training_data = training_data.select(training_data['Survived'].cast('float'),
                         training_data['Pclass'].cast('float'),
                         training_data['Sex'],
                         training_data['Age'].cast('float'),
                         training_data['Fare'].cast('float'),
                         training_data['Embarked']
                        )

#Replace None value
training_data = training_data.replace('?', None).dropna(how='any')

# Now, the Spark ML library only works with numeric data.
# But we still want to use the Sex and the Embarked column. For that,
# we will need to encode them with StringIndexer
# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
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

model = pipeline.fit(training_data)
model.save(directory_path_pipeline)

# Evaluate our model
evaluator = MulticlassClassificationEvaluator(
    labelCol='Survived',
    predictionCol='prediction',
    metricName='accuracy')

grid = ParamGridBuilder().addGrid(lr.maxIter, [500]) \
                                .addGrid(lr.regParam, [0]) \
                                .addGrid(lr.elasticNetParam, [1]) \
                                .build()
#Define the cross validator for executing the pipeline and performing cross validation.
lr_cv = CrossValidator(estimator=pipeline,
                       estimatorParamMaps=grid,
                       evaluator=evaluator,
                       numFolds=3)

# Fit the model on training data
cvModelLR = lr_cv.fit(training_data)
cvModelLR.save(directory_path_model)

# Save the best model
#bestModel = cvModelLR.bestModel

#Save Model
#bestModel.write().overwrite().save(directory_path_model)
