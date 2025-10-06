FROM python:3.11-slim

# Install system dependencies required by Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /aequitas

# Copy all necessary files for the package
COPY pyproject.toml uv.lock README.md LICENSE MANIFEST.in ./
COPY src/ ./src/
COPY serve.py ./

# Create virtual environment and install dependencies
RUN uv venv && \
    # uv sync --frozen --extra webapp --extra cli
    uv sync --frozen --extra webapp --extra cli --extra flow

# Make uv-installed Python available
ENV PATH="/aequitas/.venv/bin:$PATH"

# Set entrypoint
ENTRYPOINT ["python"]
CMD ["serve.py"]