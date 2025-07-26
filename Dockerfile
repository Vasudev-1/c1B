FROM python:3.11-slim

WORKDIR /app

COPY src /app/src/
COPY Challenge_1b /app/Challenge_1b/

RUN pip install --no-cache-dir pymupdf sentence-transformers rake-nltk nltk tqdm numpy \
 && python -m nltk.downloader punkt stopwords

# Debug: Check what files were copied
RUN ls -R /app

CMD ["python", "-m", "src.__main__", "--base", "Challenge_1b"]
