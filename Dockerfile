FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install ffmpeg (required for Whisper to process audio files)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock* .

# Install dependencies
RUN uv sync --frozen

# Download Whisper model during build
RUN uv run python -c "import whisper; whisper.load_model('base')"

# Copy application code
COPY app/ .

# Ensure LanceDB data structure is complete
# Docker COPY should preserve all directories, but verify and fix if needed
RUN if [ -d "utils/lancedb_data" ]; then \
        echo "Checking LanceDB data structure..." && \
        find utils/lancedb_data -type d -name "_versions" | wc -l && \
        echo "_versions directories found"; \
    fi

# Verify LanceDB data was copied correctly with all subdirectories
# Note: If using volume mounts (recommended), this check runs at build time
# The volume mount in docker-compose.yml will override this at runtime
RUN if [ -d "utils/lancedb_data" ]; then \
        echo "=== LanceDB data directory verification ===" && \
        versions_count=$(find utils/lancedb_data -type d -name "_versions" | wc -l) && \
        echo "Found $versions_count _versions directories" && \
        if [ "$versions_count" -eq 0 ]; then \
            echo "WARNING: No _versions directories found in image!" && \
            echo "This is OK if using volume mounts in docker-compose.yml" && \
            echo "Directory structure:" && \
            find utils/lancedb_data -type d | head -20; \
        else \
            echo "✓ _versions directories found in image" && \
            echo "Checking for manifest files:" && \
            find utils/lancedb_data -name "*.manifest" | head -5; \
        fi; \
    else \
        echo "WARNING: utils/lancedb_data directory not found in image!" && \
        echo "This is OK if using volume mounts in docker-compose.yml"; \
    fi

# Explicitly ensure LanceDB data directory is copied and verify it exists with complete structure
RUN if [ ! -d "utils/lancedb_data" ]; then \
        echo "ERROR: utils/lancedb_data directory not found!" && \
        echo "Contents of utils/:" && ls -la utils/ && \
        exit 1; \
    else \
        echo "LanceDB data directory found. Contents:" && \
        ls -la utils/lancedb_data/ && \
        echo "LanceDB tables:" && \
        find utils/lancedb_data -name "*.lance" -type d | head -10 && \
        echo "Verifying table structure..." && \
        for table_dir in utils/lancedb_data/*.lance; do \
            if [ -d "$table_dir" ]; then \
                echo "Checking table: $table_dir" && \
                ls -la "$table_dir/" && \
                if [ ! -d "$table_dir/_versions" ]; then \
                    echo "WARNING: Missing _versions directory in $table_dir" && \
                    ls -la "$table_dir/" || true; \
                else \
                    echo "✓ _versions directory exists in $table_dir" && \
                    ls -la "$table_dir/_versions/" || true; \
                fi && \
                if [ ! -d "$table_dir/data" ]; then \
                    echo "WARNING: Missing data directory in $table_dir"; \
                else \
                    echo "✓ data directory exists in $table_dir"; \
                fi; \
            fi; \
        done && \
        echo "Setting proper permissions for LanceDB data..." && \
        chmod -R u+rX utils/lancedb_data && \
        echo "Permissions set. Verifying access..." && \
        ls -la utils/lancedb_data/ && \
        echo "LanceDB data directory is ready."; \
    fi

# Use uv to run uvicorn with the correct environment
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]