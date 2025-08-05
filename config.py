import os

PROJECT_NAME = "session_1"
INPUT_DIR = "annual_report"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 100
N_PAGE_SUMMARY = 3
N_QUESTIONS = 1
API_KEY = "YOUR API KEY"
CHUNKS_DIR = os.path.join(PROJECT_NAME, "chunks")
METADATA_PATH = os.path.join(PROJECT_NAME, "metadata.json")