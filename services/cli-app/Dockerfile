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

# Configure poetry to create the venv in the project directory
RUN poetry config virtualenvs.in-project true

# Install dependencies into the virtual environment.
RUN poetry install --no-root

# Stage 2: Create the final, lean image
FROM python:3.13-slim

# Set the virtual env path
ENV VIRTUAL_ENV=/app/.venv

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Copy the application source code
COPY src/ ./src

# This entrypoint manually sets the PATH and then executes the CMD.
ENTRYPOINT ["/bin/sh", "-c", "export PATH=\"/app/.venv/bin:$PATH\" && exec \"$@\"", "--"]

# The default command to run is the python script.
CMD ["python", "-u", "src/main.py"]
