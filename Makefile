.PHONY: install test lint clean run dashboard docker-build docker-run ci-local

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.main

dashboard:
	streamlit run src/dashboard/app.py

docker-build:
	docker build -t demand-forecast .

docker-run:
	docker run -p 8501:8501 -v $(PWD)/data:/app/data demand-forecast

ci-local:
	ruff check src/ tests/
	ruff format --check src/ tests/
	pytest tests/ -v --tb=short --cov=src --cov-fail-under=80
