# syntax=docker/dockerfile:1.7
FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --requirement requirements.txt \
    && python -m pip install --no-deps .

USER 10001:10001
ENTRYPOINT ["qqdup"]
CMD ["--help"]
