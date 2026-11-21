# ----------------------------------------
# Base Image
# ----------------------------------------
FROM python:3.10-slim

# ----------------------------------------
# Working Directory
# ----------------------------------------
WORKDIR /app

# ----------------------------------------
# Environment Variables
# ----------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------------------
# Install Dependencies
# ----------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------------------
# Copy Application Source
# ----------------------------------------
COPY backend/ ./backend/

# ----------------------------------------
# Expose API Port
# ----------------------------------------
EXPOSE 8000

# ----------------------------------------
# Start FastAPI Application
# ----------------------------------------
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]