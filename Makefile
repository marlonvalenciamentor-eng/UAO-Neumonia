.PHONY: install run test docker-build docker-run clean

install:
	uv sync

run:
	uv run detector_neumonia.py

test:
	uv run pytest -v

docker-build:
	docker build -t neumonia .

docker-run:
	docker run -v $$(pwd)/data:/app/data neumonia

clean:
	rm -rf __pycache__ .pytest_cache src/__pycache__ tests/__pycache__
