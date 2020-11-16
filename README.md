# Predict Survived Passengers in Real Time

The purpose of this project is to setup a machine learning real time pipeline with Apache Spark

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- Be sure to have python3.7.
- Check the `requirements.txt` file for all the needed python packages. 

## Train the model

```
make train-model
```

## Use the model in Streaming Pipeline

```
make use-model
```

## Directories

```
├───data : our data storage directory
│   ├───input : the input data to predict
│   ├───output : the predict pipeline result
│   ├───test :  the test data to be pushed to input directory
│   └───train : the train data to use to fit our model
├───model : the directory where we store the model 
├───src : python scripts as package

```

## Authors

* ** Youssef Hissou **