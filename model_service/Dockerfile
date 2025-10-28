FROM python:3.10-slim

# Đặt thư mục làm việc
WORKDIR /code

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và các thư viện khác
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements và cài đặt thư viện Python
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Sao chép toàn bộ code của service vào image
COPY . /code/

# Mở cổng 8000
EXPOSE 8000

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]