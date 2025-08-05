import os
import json
import pandas as pd
import random
from openai import OpenAI
from config import *
from typing import List, Dict

# Initialize Perplexity client
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

def generate_question(summary, chunk):
    """Generates a single question based on a chunk and the PDF summary"""
    response = client.chat.completions.create(
        model="sonar",
        messages=[
            {
                "role": "system",
                "content": """
                You are a question generator. 
                I will provide a chunk of information along with its PDF context. 
                Your task is to generate one question with the following requirements:
                (1) The question should be based solely on the chunk’s content.
                (2) The question should include enough context from the summary (company name and year) to make it clear what the question is about.
                (3) Do not add any extra information.
                (4) If the chunk lacks useful content, respond with an empty string.
                """
            },
            {
                "role": "user",
                "content": f"PDF Summary: {summary}\nChunk Text: {chunk}\nYour question:"
            }
        ]
    )
    return response.choices[0].message.content.strip()

def run_question_generation(metadata, save = True):
    """Loops through all PDFs and generates N_QUESTIONS per file"""

    result = {
        'file_name': [],
        'format_name': [],
        'file_path': [],
        'summary': [],
        'chunk': [],
        'chunk_id': [],
        'question': []
    }

    for entry in metadata:
        file_name = entry["file_name"]
        format_name = entry["format_name"]
        file_path = entry["file_path"]
        summary = entry["summary"]
        chunk_path = os.path.join(CHUNKS_DIR, f"{format_name}.json")

        if not os.path.exists(chunk_path):
            print(f"Warning: chunk file not found for {file_name}")
            continue

        with open(chunk_path, "r") as f:
            chunks = json.load(f)

        for _ in range(N_QUESTIONS):
            chunk_id = random.randint(0, len(chunks) - 1)
            chunk = chunks[chunk_id]
            question = generate_question(summary, chunk)

            result['file_name'].append(file_name)
            result['format_name'].append(format_name)
            result['file_path'].append(file_path)
            result['summary'].append(summary)
            result['chunk'].append(chunk)
            result['chunk_id'].append(chunk_id)
            result['question'].append(question)
            
    df = pd.DataFrame(result)

    if save:
        output_path = os.path.join(PROJECT_NAME, "questions.csv")
        df.to_csv(output_path, index = False)
        print(f"Saved {len(df)} questions to {output_path}")

    return df 