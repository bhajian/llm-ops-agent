FROM python:3.11-slim

WORKDIR /app

# Install requirements before copying everything for better cache reuse
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Only copy source for prod; ignored when using volume in dev
COPY . .

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
