import json
import os
import tempfile
from http import HTTPStatus

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from starlette.responses import Response

from utils.whisper import transcribe_mp3

router = APIRouter()

@router.post("/", dependencies=[])
async def transcribe(
    file: UploadFile = File(...),
) -> Response:
    """
    Transcribe an MP3 audio file.
    
    Args:
        file: The MP3 file to transcribe
        
    Returns:
        JSON response with the transcription
    """
    # Create a temporary file to save the uploaded MP3
    print(f"File: {file}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        try:
            # Save the uploaded file to the temporary location
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            # Transcribe the MP3 file
            transcription = transcribe_mp3(tmp_file_path)
            
            return Response(
                content=json.dumps({"transcription": transcription}),
                status_code=HTTPStatus.OK,
            )
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)   