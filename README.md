# decision-intelligence-assistant

## Project Overview

This project is an end-to-end AI system designed to analyze customer support queries, classify their urgency, and generate helpful responses using both machine learning and large language models (LLMs).

The system combines:
- A machine learning model for priority prediction
- A Retrieval-Augmented Generation (RAG) pipeline using a vector database (Qdrant)
- A non-RAG LLM baseline for comparison
- A full-stack deployment using FastAPI, React, and Docker

The goal is to explore the tradeoffs between traditional ML models, RAG-based systems, and pure LLM approaches in a realistic customer support scenario.

---

## Problem Statement

Customer support systems must handle large volumes of user queries efficiently. A key challenge is determining:

- Which queries are urgent  
- How to generate helpful and accurate responses  
- How to balance speed, cost, and quality  

This project simulates a production scenario where thousands of support tickets must be processed, and compares different approaches to solving this problem.

---

## Dataset

Dataset used:
**Customer Support on Twitter (Kaggle)**

### Key characteristics:
- Noisy and unstructured text  
- Mostly user complaints (not structured solutions)  
- No explicit urgency labels  

### Labeling approach:
Weak supervision was used based on keywords such as:
- "urgent"
- "help"
- "asap"
- excessive punctuation


## System Architecture

User Query → Feature Extraction → ML Model → Qdrant → LLM (RAG + Non-RAG) → Frontend

### Components:
- Backend: FastAPI  
- Frontend: React (Vite)  
- Vector Database: Qdrant  
- LLM: OpenAI GPT-4.1-mini  
- Deployment: Docker + Docker Compose  


## Features

- Priority classification using a trained ML model  
- Zero-shot classification using an LLM  
- RAG-based answer generation using Qdrant  
- Non-RAG answer generation for comparison  
- Logging of:
  - user queries  
  - retrieved documents  
  - predictions  
  - latency  
  - cost estimates  
- Interactive frontend dashboard  



