.PHONY: install run test docker-build docker-run clean

install:
	pip install -r requirements.txt

run:
	python detector_neumonia.py

test:
	pytest -v

docker-build:
	docker build -t neumonia .

docker-run:
	docker run -v $$(pwd)/data:/app/data neumonia

clean:
	rm -rf __pycache__ .pytest_cache
