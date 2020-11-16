.DEFAULT: help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "---------------HELP-----------------"
	@echo "make clean-environment"
	@echo "       Clean the Data/Model environment"
	@echo "make train-model"
	@echo "       train the model using spark ml"
	@echo "make use-model"
	@echo "       use the model using spark streaming "
	@echo "------------------------------------"


clean-environment: ##Clean the Data/Model environment
	python ./src/clean_environment.py

train-model: ##Train the model
			./venv/bin/spark-submit src/train_model.py

use-model: ##Use the model in spark Streaming
			./venv/bin/spark-submit src/use_model.py
