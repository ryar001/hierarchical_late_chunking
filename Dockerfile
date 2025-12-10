FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# build-essential for compiling some python packages
# netcat-traditional for health checking script (optional but good practice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy project definition
COPY pyproject.toml .

# Install dependencies using uv
# We use --system to install into the container's global environment
# We compile to requirements.txt first to leverage Docker caching for dependencies
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --system -r requirements.txt

# Copy source code
COPY . .

# Install the project itself
RUN uv pip install --system --no-deps .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port (Chainlit default)
EXPOSE 8000

# Command to run the application
# We use shell form to allow variable expansion if needed, but array form is safer for signals.
# Chainlit needs to bind to 0.0.0.0 to be accessible from outside the container.
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
