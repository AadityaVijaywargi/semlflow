.PHONY: install lint test train ui docker-ui clean

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

test:
	pytest -q

# full DVC pipeline (prepare -> train_ml -> train_dl -> evaluate), logged to MLflow
train:
	bash scripts/run_with_mlflow.sh dvc repro

# open the MLflow web GUI at http://127.0.0.1:5000
ui:
	mlflow server --host 127.0.0.1 --port 5000 \
	  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

docker-ui:
	docker build -f Dockerfile.ui -t mlflow-ui .
	docker run --rm -p 5000:5000 mlflow-ui

clean:
	rm -rf mlflow.db mlartifacts mlruns mlflow-server.log data/raw data/processed models metrics reports
