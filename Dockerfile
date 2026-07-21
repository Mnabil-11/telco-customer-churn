FROM python:3.11-slim

WORKDIR /code

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ app/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
