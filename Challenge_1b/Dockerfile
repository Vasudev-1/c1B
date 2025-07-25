FROM python:3.10-slim
WORKDIR /app
COPY ../src/ ./src
COPY . ./Challenge_1b
RUN pip install --no-cache-dir pymupdf sentence-transformers rake-nltk nltk tqdm numpy
CMD ["python", "src/__main__.py", "--base", "Challenge_1b"]
