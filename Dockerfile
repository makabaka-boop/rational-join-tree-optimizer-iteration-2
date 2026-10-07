# syntax=docker/dockerfile:1

# The planner itself: pure Python stdlib, no database, no dependencies.
FROM python:3.12-slim AS app
WORKDIR /app
COPY joinplan.py ./
ENTRYPOINT ["python", "joinplan.py"]

# Test image: same code plus pytest and the test suite.
FROM app AS tests
RUN pip install --no-cache-dir pytest
COPY test_joinplan.py ./
ENTRYPOINT ["python", "-m", "pytest", "-q"]
