import whisper

def transcribe_mp3(mp3_file_path, model_name="base"):
    """
    Transcribes an MP3 audio file using OpenAI's Whisper model.

    Args:
        mp3_file_path (str): Path to the mp3 file to transcribe.
        model_name (str): Whisper model size to use, e.g., "tiny", "base", "small", "medium", "large".

    Returns:
        str: The transcription of the audio.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(mp3_file_path)
    print(f"Result: {result}")
    return result["text"]


