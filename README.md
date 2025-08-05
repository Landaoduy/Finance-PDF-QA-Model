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
