import os
import logging
import sys

# Set TOKENIZERS_PARALLELISM to avoid warnings when uvicorn forks processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging to output to stdout/stderr for Docker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info("Starting FastAPI application...")

from fastapi import FastAPI
from router import router as process_router
import seed

seed.seedall()

app = FastAPI()
app.include_router(process_router)
logger.info("FastAPI application initialized successfully")
