# -------- Base Image --------
FROM python:3.10-slim

# -------- Environment --------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------- Set Work Directory --------
WORKDIR /app

# -------- Install System Dependencies --------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------- Copy Requirements --------
COPY requirements.txt .

# -------- Install Python Dependencies --------
RUN pip install --no-cache-dir -r requirements.txt

# -------- Copy Application Code --------
COPY . .

# -------- Expose Port --------
EXPOSE 8000

# -------- Run Server --------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]