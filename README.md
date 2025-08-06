![UTA-DataScience-Logo](https://github.com/user-attachments/assets/6d626bcc-5430-4356-927b-97764939109d)

# Finance PDF Q&A Model
* This project builds an end-to-end LLM-powered system that answers questions about financial reports (PDFs). Users can either **auto-generate questions** from the uploaded document or **manually type their own**, then receive context-aware answers based on document retrieval.

* To assess answer quality, the system includes an **LLM-based evaluation module** that scores each response across three dimensions: **factual correctness**, **completeness**, and **clarity**. These scores are generated entirely by the language model using a structured evaluation rubric.

* **Key Components**:
  * **LangChain** – framework for managing chunking, document embedding, retrieval, and chaining components together
  * **Perplexity's Sonar LLM** – used for question generation, answer generation, and evaluation
  * **FAISS** – enables fast vector similarity search over document chunks
  * **Streamlit** – provides an interactive web app for user input and result visualization
  * **Plotly** – used to generate evaluation metric visualizations (box plots, histograms, heatmaps)

## Overview
* Financial documents are often dense, multi-page, and highly structured. Extracting useful insights through LLMs presents challenges in chunking, hallucination prevention, and evaluation.

* This project simulates a realistic pipeline where:
  * Annual reports are broken into chunks and summarized
  * Questions are auto-generated using **Perplexity Sonar**
  * A retrieval-based QA model answers these questions
  * Each answer is scored by the LLM evaluator using a 5-point rubric on Factual Correctness, Completeness, and Clarity

## Repository Structure
```sh
├── app.py                     # Streamlit web app for uploading and interacting with PDFs
├── config.py                  # Project config and global constants
├── data_processing.py         # PDF loading, chunking, and summarization
├── question_generator.py      # LLM-based question generation from chunks
├── retriever_model.py         # Embedding, FAISS index creation, retrieval pipeline
├── evaluation.py              # Auto-evaluation using rubric-based prompts
├── visualize.py               # Plotly visualizations for evaluation metrics
├── main.ipynb                 # End-to-end notebook running the full pipeline
├── annual_report/             # Sample input folder containing financial PDFs
├── session_1/                 # Output folder: metadata, questions, answers, evaluated CSVs
├── faiss_index_open/          # Saved FAISS vector store
```
## Summary of Workdone

### Data
* **Input**:
  * PDF financial reports from 10 different companies placed in annual_report/
  * Each file is processed independently

* **Output**:
  * Summary of first 3 pages
  * List of chunked text segments (JSON)
  * Auto-generated questions (questions.csv)
  * Model-generated answers (answers.csv)
  * Evaluation scores and comments (evaluated.csv)
  
