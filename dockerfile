FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Tạo thư mục artifacts
RUN mkdir -p /app/artifacts

# (Các lệnh gdown hoặc COPY artifacts của bạn)
# ...

COPY ./app /app/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]