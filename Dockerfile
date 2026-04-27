FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('wordnet')"
COPY . .
RUN mkdir -p quora-question-pairs
CMD ["python", "Inference_Quora_Contest.py"]