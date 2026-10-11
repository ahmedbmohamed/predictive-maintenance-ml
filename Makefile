.PHONY: install train test lint api dashboard docker

install:
	pip install -r requirements-dev.txt && pip install -e .

train:
	python -m pdm.train

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run dashboard/app.py

docker:
	docker compose up --build
