# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .

RUN pip install -e . --no-cache-dir --no-deps

VOLUME ["/app/data", "/app/output", "/app/models"]
EXPOSE 5000

ENTRYPOINT ["python", "main.py"]
CMD ["train"]
