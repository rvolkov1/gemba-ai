# Stage 1: Build the virtual environment with Poetry
FROM python:3.13-slim AS builder

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Install poetry
RUN python -m venv $POETRY_HOME
RUN $POETRY_HOME/bin/pip install poetry


# Copy dependency definition files
COPY pyproject.toml poetry.lock poetry.toml ./

# Install dependencies into the virtual environment. 
# --no-root: Don't install the project itself, only dependencies.
# --no-dev: Exclude development dependencies.
RUN poetry install --no-root

# Pre-download RF-DETR Nano model weights
RUN python -c "from rfdetr import RFDETRNano; model = RFDETRNano()"

# Stage 2: Create the final, lean image
FROM python:3.13-slim

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Copy the application source code
COPY src/ ./src

# The command to run the FastAPI application using uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
