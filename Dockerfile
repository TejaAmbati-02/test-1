# syntax=docker/dockerfile:1

############################
# Stage 1 — builder
############################
# Compiles any dependencies that ship only as source (e.g. oci / cryptography /
# opensearch-py on a fresh CPython) into a self-contained virtualenv. Build tools
# live only here and never reach the runtime image.
FROM python:3.14.8-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv we can copy wholesale into the runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt


############################
# Stage 2 — runtime
############################
FROM python:3.14.8-slim AS runtime

# - PYTHONUNBUFFERED: stream logs straight to stdout (no buffering)
# - PYTHONDONTWRITEBYTECODE: no .pyc files at runtime
# - PATH: use the pre-built venv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# tini = lightweight init as PID 1: forwards signals and reaps zombies so the
# container shuts down cleanly (triggers the lifespan heartbeat cancel + client close).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring the fully-built dependencies over from the builder (no compilers in this image).
COPY --from=builder /opt/venv /opt/venv

# Copy application code (see .dockerignore for what is excluded).
COPY . .

# Non-root user. OCI config is mounted into this user's home at runtime.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container healthcheck against the app's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["tini", "--"]

# Production run: no auto-reload.
# --proxy-headers + --forwarded-allow-ips: trust the upstream LB so the request logger
#   records the real client IP (X-Forwarded-For) instead of the proxy's address.
# Scale by running more container replicas rather than many in-process workers — each
# worker holds its own OpenSearch/OCI connection pool and heartbeat.
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--no-server-header"]
