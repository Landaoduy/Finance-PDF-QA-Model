from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from config import *
import shutil
import json

# Initialize Perplexity client
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

def setup_folders():
    # Delete old session (for dev)
    if os.path.exists(PROJECT_NAME):
        shutil.rmtree(PROJECT_NAME)
        print(f"Deleted existing folder: {PROJECT_NAME}")

    os.makedirs(CHUNKS_DIR, exist_ok = True)
    print(f"Created folders: {PROJECT_NAME}, {CHUNKS_DIR}")

def extract_summary(first_n_pages):
    """Uses Perplexity (Sonar) to generate a short summary for first N pages"""
    response = client.chat.completions.create(
        model="sonar",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
                You are a financial report assistant.
                I will provide the first few pages of a financial report, and your task is to give a concise, single-sentence summary answering:
                (1) which company the report is about and 
                (2) what year it covers. 
                Limit the summary to 50 words, with no extra details or formatting.
                """
            },
            {
                "role": "user",
                "content": f"""
                The first few pages: {first_n_pages}
                Your response:
                """
            },
        ]
    )
    return response.choices[0].message.content.strip()  # Fix: make sure it's at correct indentation level

def process_pdfs():
    """Process all PDFs in the input folder, return updated metadata list"""
    # Load existing metadata if it exists
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    else:
        metadata = []
        
    existing_files = {entry["file_name"] for entry in metadata}

    for file_name in os.listdir(INPUT_DIR):
        if not file_name.endswith(".pdf") or file_name in existing_files:
            continue 
        
        file_path = os.path.join(INPUT_DIR, file_name)
        format_name = file_name.rsplit(".", 1)[0]

        print(f"Processing: {file_name}")

        # Load PDF
        loader = PyMuPDFLoader(file_path)
        documents = loader.load() 

        # Count words 
        word_count = sum(doc.page_content.count(" ") for doc in documents)

        # Extract summary from first N pages
        first_n_pages = "\n".join(doc.page_content for doc in documents[:N_PAGE_SUMMARY])
        summary = extract_summary(first_n_pages)

         # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(documents)
        chunk_count = len(chunks)

        # Save chunks as JSON
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_path = os.path.join(CHUNKS_DIR, f"{format_name}.json")
        with open(chunk_path, "w") as f:
            json.dump(chunk_texts, f, indent=2)

        # Update metadata
        metadata.append({
            "file_name": file_name,
            "format_name": format_name,
            "file_path": file_path,
            "chunk_count": chunk_count,
            "total_word_count": word_count,
            "summary": summary
        }) 

        print(f"Finished {file_name}: {chunk_count} chunks, {word_count} words.")

    # Save updated metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata   