# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 PYTHONPATH=/app HF_HOME=/tmp/hf
WORKDIR /app

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# The frozen code resolves paths relative to the package's parent (ROOT=/app):
#   /app/polyjudge  /app/splits  /app/cache (replay cache)  /app/data/test_input
COPY src/polyjudge ./polyjudge
COPY data/splits ./splits
COPY data/test_input ./data/test_input
COPY data/few_shot ./data/few_shot
COPY scripts ./scripts
# Replay cache: M1 entries are in data/replay_cache/, M0/M2/M3 entries in data/replay_cache/extras/.
# Keys are content hashes and disjoint between methods, so both are merged into the single dir the code reads.
RUN --mount=type=bind,source=data/replay_cache,target=/rc \
    mkdir -p /app/cache \
 && cp -r /rc/[0-9a-f][0-9a-f] /app/cache/ \
 && cp -r /rc/extras/[0-9a-f][0-9a-f] /app/cache/

# ---- tests (no organizers data needed) ----
FROM base AS test
COPY requirements.lock requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests ./tests
CMD ["python", "-m", "pytest", "-q", "tests"]

# ---- eval: the organizers' repo is cloned at build time, pinned; never vendored into this repo ----
# Heavy (their requirements pull torch & co.) and network is needed at BUILD time only.
FROM base AS eval
ARG ORGANIZERS_COMMIT=1807a42100e742ed03d337c54c4b9ea86995f565
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates  && rm -rf /var/lib/apt/lists/*
RUN git clone --quiet https://github.com/ZurichNLP/SwissGov-RSD /opt/organizers_repo  && git -C /opt/organizers_repo checkout --quiet ${ORGANIZERS_COMMIT}
RUN pip install --no-cache-dir -r /opt/organizers_repo/requirements.txt
ENV ORGANIZERS_REPO=/opt/organizers_repo PYTHONPATH=/opt/organizers_repo:/app
WORKDIR /opt/organizers_repo

# ---- runtime image: LAST stage on purpose, so a plain `docker build .` builds this one (not eval) ----
FROM base AS runtime
CMD ["python", "-m", "polyjudge.predict", "--help"]
