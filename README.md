Training: รันคำสั่ง python -m src.train

Serving (API): รันคำสั่ง uvicorn api.app:app --reload

Docker Build: docker build -t pinn-option .

Docker Run: docker run -p 8000:8000 pinn-option