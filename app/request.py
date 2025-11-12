import requests
import json
import uuid
import os


def send_test_event():
    # API endpoint URL (adjust if your server runs on a different port)
    url = "http://localhost:8000/events/"

    # Sample event data
    event_data = {
        "event_id": str(uuid.uuid4()),  # Generate a random UUID
        "event_type": "test_event",
        "event_data": {
            "message": "Can you explain how to use FastAPI?",
        },
    }

    # Headers for JSON content
    headers = {
        "Content-Type": "application/json",
    }

    # Send POST request to the endpoint
    response = requests.post(url=url, data=json.dumps(event_data), headers=headers)

    # Print response information
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"Full Response Headers: {response.headers}")



def send_transcribe_request():
    url = "http://localhost:8000/transcribe/"
    # Read the file as binary
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "DEDE001.mp3")
    print(f"File path: {file_path}")
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(url=url, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print(f"Full Response Headers: {response.headers}")

if __name__ == "__main__":
    # send_test_event()
    send_transcribe_request()
