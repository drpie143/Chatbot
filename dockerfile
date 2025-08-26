# Sử dụng image nền có sẵn Python
FROM python:3.9-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Chạy lệnh pip để cài đặt tất cả các thư viện
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Cài đặt công cụ tải file từ Google Drive
RUN pip install gdown

# Tạo thư mục artifacts
RUN mkdir -p /app/artifacts

# <<< SỬA ĐỔI TẠI ĐÂY >>>
# Tải file id_to_record.pkl từ Google Drive
# !!! THAY <ID_FILE_CỦA_BẠN> BẰNG ID THẬT CỦA BẠN !!!
RUN gdown --id 1MR6wg2wP1Qc0g9EgCLAWkNkgZoNVAZYJ -O /app/artifacts/id_to_record.pkl

# Sao chép code API vào
COPY ./app /app/

# <<< THÊM DÒNG NÀY ĐỂ SỬA LỖI CỔNG >>>
# Báo cho Docker biết rằng ứng dụng sẽ lắng nghe trên cổng 8000
EXPOSE 8000

# Lệnh khởi động ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]