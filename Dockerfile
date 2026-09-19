FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip and install dependencies with longer timeout/retries
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]