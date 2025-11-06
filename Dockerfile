FROM python:3.10

# Install system dependencies required by Python packages
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     g++ \
#     cmake \
#     pkg-config \
#     libcairo2-dev \
#     libpango1.0-dev \
#     libffi-dev \
#     shared-mime-info \
#     && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY . /aequitas
WORKDIR /aequitas

# Create virtual environment and install dependencies
RUN uv venv && \
    uv sync --extra webapp --extra cli
    # uv sync --frozen --extra webapp --extra cli --extra flow

# Make uv-installed Python available
ENV PATH="/aequitas/.venv/bin:$PATH"

# Set entrypoint
ENTRYPOINT ["python"]
CMD ["serve.py"]