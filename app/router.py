from fastapi import APIRouter

import endpoint
import transcribe

router = APIRouter()

router.include_router(endpoint.router, prefix="/events", tags=["events"])
router.include_router(transcribe.router, prefix="/transcribe", tags=["transcribe"])