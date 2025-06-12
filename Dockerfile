# llm-ops-agent/Dockerfile  (agent-server)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# ADD --verbose HERE to see detailed installation logs
RUN pip install --no-cache-dir --verbose -r requirements.txt

# copy *entire* repo so `app/` package is present inside container
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
