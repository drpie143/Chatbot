# Bước 1: Chọn "Nền" (Base Image)
# Bắt đầu với một phiên bản Linux tối giản đã được cài sẵn Python 3.9.
FROM python:3.9-slim

# Bước 2: Tạo không gian làm việc
# Tạo một thư mục tên là /app bên trong "chiếc hộp" (container) và di chuyển vào đó.
WORKDIR /app

# Bước 3: Sao chép file "danh sách mua sắm"
# Chép file requirements.txt từ máy bạn vào thư mục /app trong container.
COPY requirements.txt .

# Bước 4: Cài đặt các "nguyên liệu" cần thiết
# Chạy lệnh pip để cài đặt tất cả các thư viện được liệt kê trong requirements.txt.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Bước 5: Sao chép "nguyên liệu" chính
# Chép toàn bộ thư mục 'artifacts' (chứa index) và 'app' (chứa code) từ máy bạn vào container.
COPY ./artifacts /app/artifacts
COPY ./app /app/

# Bước 6: Chỉ định cách "dọn món" (khởi động ứng dụng)
# Đây là lệnh sẽ được chạy khi container khởi động. 
# Lệnh này yêu cầu uvicorn chạy ứng dụng FastAPI trong file main.py của bạn.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]