# RAG Model based on Dostoevskiy's books 
## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) model designed to answer questions about the works of Fyodor Dostoevsky, including novels like "Crime and Punishment", "The Idiot", "The Brothers Karamazov", "White nights" and "Short Stories".

## Project Structure
- `data/`: Additional data and reference materials.
- `test/`: Directory containing the golden set of questions and answers.
  - `goldset.txt`: Contains the original set of questions and expected answers.
  - `test_and_precision.pdf`: Contains model answers and calculation of P@10.
- `vector_db_dir/`: Contains vectorized books.

## Model Description
The RAG model combines:
- Information retrieval from a pre-defined knowledge base
- Generative AI techniques to formulate precise answers
- Specialized focus on Dostoevsky's literary works

## Performance Metrics
- **Precision@10**: 0.8 (80% accuracy for top 10 questions)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python -m streamlit run app.py
```

## Limitations
- Accuracy depends on the comprehensiveness of the knowledge base
- May struggle with highly nuanced or interpretative questions