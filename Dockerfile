FROM python:3.11-sli

# Install system dependencies required for compiling psycopg2 if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Copy dependency files from root
COPY pyproject.toml uv.lock ./

# Install project dependencies globally into the container system
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy scripts
COPY . .

# Gradio default port
EXPOSE 8080

# Run main.py located inside the app/ directory
CMD ["python", "app/main.py"]