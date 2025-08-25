# (Các lệnh khác giữ nguyên)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt công cụ tải file từ Google Drive
RUN pip install gdown

# Tạo thư mục artifacts trước khi tải file vào đó
RUN mkdir -p /app/artifacts

# Lệnh tải các file lớn từ Google Drive
# !!! THAY CÁC URL DƯỚI ĐÂY BẰNG LINK TỪ GOOGLE DRIVE CỦA BẠN !!!
RUN gdown --id 1GNlkFsfdIUwnXxVad4wN8XUAJ4ut38r1 -O /app/artifacts/index.faiss
RUN gdown --id 1ehKN4_IMk-YOv3BTXLPBMDv1tGmBahV2 -O /app/artifacts/metas.pkl.gz
RUN gdown --id 1UImHA5i8OkspDfK54XRNDboA24pcMn4- -O /app/artifacts/bm25.pkl.gz
RUN gdown --id 1MR6wg2wP1Qc0g9EgCLAWkNkgZoNVAZYJ -O /app/artifacts/id_to_record.pkl

# (Các lệnh khác giữ nguyên)
COPY ./app /app/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
