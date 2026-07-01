# YouTube Intelligence Engine

NLP pipeline that scrapes YouTube comments and transcripts from videos about "AI replacing jobs" and builds a RAG system on top to answer questions about public sentiment.

CSCI370 (NLP) project, Spring 2026.

## What it does

Scrapes around 27,000 comments from 8 videos, runs them through sentiment analysis, topic modeling, NER, and entity extraction, then loads everything into ChromaDB so an LLM agent can answer questions using the actual comment data instead of guessing.

The main finding: the dataset is about 50% negative sentiment, but that's because of which videos got picked, not because that reflects real public opinion. The dashboard has a page that proves this directly — same question, but filtered to negative vs positive comments, gives two completely different answers.

## Stack

- Scraping: YouTube Data API v3, `youtube-transcript-api`
- NLP: spaCy, RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`), BERTopic, KeyBERT, GLiNER
- RAG: ChromaDB (3 collections), LangChain, LangGraph, BM25 lexical retrieval
- LLM: Groq (`llama-3.3-70b-versatile`)
- Tracking: MLflow
- Dashboard: Streamlit

## Setup

```bash
git clone https://github.com/amiresTkm/youtube-intelligence-engine.git
cd youtube-intelligence-engine

py -3.11 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

This needs Python 3.11 specifically. 3.12 and 3.13 currently break ChromaDB and BERTopic.

GPU support is optional but speeds up sentiment analysis and embeddings significantly:
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
This needs torch 2.6 or higher — the transformers library requires this version for a security fix.

Create a `.env` file with:
YOUTUBE_API_KEY=...
GROQ_API_KEY=...

Get a YouTube key from Google Cloud Console (enable "YouTube Data API v3", choose the public data option). Get a Groq key from console.groq.com — it's free.

## Running it

Run the notebooks in this order. Each one depends on the output of the one before it:

1. `01_scraping.ipynb`
2. `02_preprocessing.ipynb`
3. `03_sentiment.ipynb`
4. `04_topic_modeling.ipynb`
5. `02b_ner_keywords.ipynb`
6. `02c_knowledge_graph.ipynb` (optional, only used for the report)
7. `05_rag_pipeline.ipynb`
8. `06_agent.ipynb`
9. `07_evaluation.ipynb`

Note: the numbering looks out of order because `02b` and `02c` actually run after `04`. They need the topic labels that notebook 4 produces. They're named that way because they belong conceptually with preprocessing, not because of when they run.

The dashboard includes automatic query analysis that routes each question to the most appropriate retrieval strategy based on detected intent.

Then start the dashboard:
```bash
streamlit run app.py
```

To view experiment tracking:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Dataset

8 YouTube videos covering different angles on AI and jobs — some pessimistic, some optimistic, some focused on economics (capitalism, UBI) and some on developers specifically. About 27,330 comments after cleaning, plus 8 full video transcripts.

## Results

- Sentiment: 50.2% negative, 31.2% neutral, 18.7% positive
- 92 topics found with BERTopic (outlier rate dropped from 45.6% to 0.3% after tuning)
- 1,113 unique entities extracted
- Average retrieval relevance: 0.79 out of 1.0
- Retrieval strategies implemented: Semantic, MMR, Metadata-filtered (sentiment/entity), BM25 lexical, Hybrid (multi-collection)
- MMR retrieval is about 9.6% more diverse than plain semantic retrieval
- LLM-as-judge scores: 5/5 faithfulness, 5/5 relevance, 4.4/5 completeness on test questions. Also tested the judge against a deliberately wrong, made-up answer to make sure it actually catches mistakes — it correctly scored that one 1/5.

## Known issues

- Groq's tool calling API sometimes fails with LangGraph (`400 tool_use_failed` error). This is a known issue that affects other projects too, not specific to this one. When it happens, the system falls back to a direct RAG answer instead of failing completely.
- spaCy sometimes mislabels AI-related acronyms and YouTube channel names as organizations or locations. A custom entity list helps but doesn't fix every case.
- The negative sentiment skew comes from which videos were chosen for the dataset, not from real public opinion. This is explained and demonstrated directly in the Bias Explorer page of the dashboard.
- Knowledge graph relation extraction is rule-based, so some of the extracted relationships are genuinely useful and some are just noise from how the text was parsed.
- The LLM-as-judge evaluation uses the same model for generating and grading answers, which can make it overly generous. To check this, I tested it with a fabricated wrong answer and it correctly gave it a low score, which gives some confidence the scoring isn't just automatically high.

## Author

Amir — CSCI370, Spring 2026