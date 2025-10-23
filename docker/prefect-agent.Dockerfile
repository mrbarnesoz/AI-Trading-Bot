FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PREFECT_HOME=/opt/prefect

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir "prefect>=2.14"

EXPOSE 4200

ENTRYPOINT ["prefect", "agent", "start", "--work-queue", "bitmex"]
