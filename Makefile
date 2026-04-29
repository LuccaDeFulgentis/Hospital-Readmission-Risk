PYTHON = $(shell if [ -f ./checkin_env/bin/python3 ]; then echo "./checkin_env/bin/python3"; else echo "python"; fi)
PYTEST = $(shell if [ -f ./checkin_env/bin/pytest ]; then echo "./checkin_env/bin/pytest"; else echo "pytest"; fi)

install:
	pip install -r requirements.txt

run:
	$(PYTHON) src/train.py

test:
	$(PYTEST) tests/