FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/etaf_agent

RUN pip install --upgrade pip && \
    pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

ENTRYPOINT ["etaf-agent"]
CMD ["--help"]
