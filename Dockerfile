# Root submission Dockerfile for the Crossing Challenge.
# Builds from the repository root, as requested by the take-home prompt.

FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by xgboost at runtime on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py grade.py model.pkl ./
COPY crossing-challenge-starter/predict.py ./crossing-challenge-starter/predict.py
COPY crossing-challenge-starter/grade.py ./crossing-challenge-starter/grade.py

# Grader invokes: python grade.py <input.parquet> <output.csv>
ENTRYPOINT ["python", "grade.py"]
