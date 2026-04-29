install:
	pip install -r requirements.txt

run:
	./checkin_env/bin/python3 src/train.py

test:
	./checkin_env/bin/pytest tests/