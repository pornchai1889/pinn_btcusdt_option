# ใช้ Python Official Image
FROM python:3.9-slim

# ตั้ง Folder ทำงาน
WORKDIR /app

# ก๊อปปี้ไฟล์โปรเจกต์เข้าไป
COPY . /app

# ติดตั้ง Library
RUN pip install --no-cache-dir -r requirements.txt

# เปิด Port 8000
EXPOSE 8000

# คำสั่งรัน API เมื่อ Container เริ่มทำงาน
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]