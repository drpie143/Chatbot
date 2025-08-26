FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Chỉ cần copy file tra cứu nhanh
COPY ./artifacts/id_to_record.pkl /app/artifacts/id_to_record.pkl

COPY ./app /app/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]