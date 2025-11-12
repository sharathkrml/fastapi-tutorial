import json
import os
import tempfile
from http import HTTPStatus

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from starlette.responses import Response

from utils.whisper import transcribe_mp3


router = APIRouter()

data_store = []

class EventSchema(BaseModel):
    """Event Schema"""

    event_id: str
    event_type: str
    event_data: dict


"""
Becuase of the router, every endpoint in this file is prefixed with /events/
"""


@router.post("/", dependencies=[])
def handle_event(
    data: EventSchema,
) -> Response:
    print(data)
    data_store.append(data)
    # This is where you implement the AI logic to handle the event

    # Return acceptance response
    return Response(
        content=json.dumps({"message": "Data received!", "data": [item.model_dump() for item in data_store]}),
        status_code=HTTPStatus.ACCEPTED,
    )

@router.get("/", dependencies=[])
def get_data() -> Response:
    return Response(
        content=json.dumps({"data": [item.model_dump() for item in data_store]}),
        status_code=HTTPStatus.OK,
    )
