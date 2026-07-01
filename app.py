import os
import ast
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from collections import Counter

import mlflow
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("youtube-intelligence-engine-dashboard")
mlflow.langchain.autolog()

# ── page config - must be first streamlit call ──
st.set_page_config(
    page_title="YouTube Intelligence Engine",
    page_icon="🎯",
    layout="wide"
)

# ── load env and initialize models ──
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_models():
    """Cache models so they don't reload on every interaction."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.3)
    chroma_client = chromadb.PersistentClient(path="data/processed/chromadb")
    comments_col = chroma_client.get_collection("comments")
    transcripts_col = chroma_client.get_collection("transcripts")
    summaries_col = chroma_client.get_collection("topic_summaries")
    return embedder, llm, comments_col, transcripts_col, summaries_col

@st.cache_data
def load_data():
    """Cache dataset so it doesn't reload on every interaction."""
    df = pd.read_csv("data/processed/comments_enriched.csv")
    df["entities_parsed"] = df["entities"].apply(ast.literal_eval)
    df["keywords_parsed"] = df["keywords"].apply(ast.literal_eval)
    return df

@st.cache_resource
def load_bm25():
    """Cache BM25 index at startup - builds once, stays in memory."""
    from rank_bm25 import BM25Okapi
    df_temp = pd.read_csv("data/processed/comments_enriched.csv")
    texts = df_temp["text"].tolist()
    index = BM25Okapi([t.lower().split() for t in texts])
    return index, texts

bm25_index, bm25_texts = load_bm25()

embedder, llm, comments_col, transcripts_col, summaries_col = load_models()
df = load_data()


# ── retrieval functions ──
def semantic_retrieval(query, collection, k=5, filter_dict=None):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=filter_dict
    )
    return results["documents"][0], results["metadatas"][0]

def mmr_retrieval(query, k=5, fetch_k=20):
    query_embedding = embedder.encode(query).tolist()
    results = comments_col.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "embeddings"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    embeddings_list = np.array(results["embeddings"][0])
    q_emb = np.array(query_embedding)

    selected = []
    remaining = list(range(len(docs)))

    for _ in range(min(k, len(docs))):
        if not remaining:
            break
        if not selected:
            scores = embeddings_list[remaining] @ q_emb
            best = remaining[int(scores.argmax())]
        else:
            rel_scores = embeddings_list[remaining] @ q_emb
            sel_embs = embeddings_list[selected]
            div_scores = (embeddings_list[remaining] @ sel_embs.T).max(axis=1)
            combined = 0.7 * rel_scores - 0.3 * div_scores
            best = remaining[int(combined.argmax())]
        selected.append(best)
        remaining.remove(best)

    return [docs[i] for i in selected], [metas[i] for i in selected]

def hybrid_retrieval(query, k_each=3):
    comment_docs, comment_meta = semantic_retrieval(query, comments_col, k=k_each)
    transcript_docs, transcript_meta = semantic_retrieval(query, transcripts_col, k=k_each)
    summary_docs, summary_meta = semantic_retrieval(query, summaries_col, k=2)
    return comment_docs, comment_meta, transcript_docs, transcript_meta, summary_docs

def generate_answer(question, comment_docs, transcript_docs, summary_docs):
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI analyst for the YouTube Intelligence Engine.
Answer questions based on real YouTube comments and video transcripts.
Always ground your answers in the provided context.
Never make up opinions not present in the context."""),
        ("human", """Question: {question}

Comments:
{comments}

Transcripts:
{transcripts}

Topic Summaries:
{summaries}

Provide a comprehensive answer based on the context above.""")
    ])
    chain = qa_prompt | llm
    response = chain.invoke({
        "question": question,
        "comments": "\n\n".join(comment_docs),
        "transcripts": "\n\n".join(transcript_docs),
        "summaries": "\n\n".join(summary_docs)
    })
    return response.content


# ── sidebar navigation ──
st.sidebar.title("🎯 YouTube Intelligence Engine")
st.sidebar.markdown("*AI & Future of Work Analysis*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dataset Overview", "💬 Ask the System", "⚖️ Bias Explorer", "🔍 Topic & Entity Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df):,} comments")
st.sidebar.markdown(f"**Videos:** {df['video_name'].nunique()}")
st.sidebar.markdown(f"**Topics:** {df['topic'].nunique()}")

# ══════════════════════════════════════════
# PAGE 1 — DATASET OVERVIEW
# ══════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("27,330 YouTube comments across 8 videos on AI & the future of work.")

    # top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Comments", f"{len(df):,}")
    col2.metric("Negative Sentiment", "50.2%")
    col3.metric("Topics Found", f"{df['topic'].nunique()}")
    col4.metric("Unique Entities", "1,113")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment Distribution")
        sent_counts = df["sentiment"].value_counts()
        fig = px.bar(
            x=sent_counts.index,
            y=sent_counts.values,
            color=sent_counts.index,
            color_discrete_map={
                "negative": "#e74c3c",
                "neutral": "#95a5a6",
                "positive": "#2ecc71"
            },
            labels={"x": "Sentiment", "y": "Count"}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Comments per Video")
        video_counts = df["video_name"].value_counts()
        fig2 = px.bar(
            x=video_counts.values,
            y=video_counts.index,
            orientation="h",
            color=video_counts.values,
            color_continuous_scale="Blues"
        )
        fig2.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Sentiment per Video")
    video_sentiment = df.groupby(["video_name", "sentiment"]).size().unstack(fill_value=0)
    video_sentiment_pct = video_sentiment.div(video_sentiment.sum(axis=1), axis=0) * 100
    fig3 = px.bar(
        video_sentiment_pct,
        barmode="stack",
        color_discrete_map={
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
            "positive": "#2ecc71"
        }
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col_kw, col_ent = st.columns(2)

    with col_kw:
        st.subheader("Top Keywords")
        all_keywords = [kw for kws in df["keywords_parsed"] for kw in kws]
        kw_counts = Counter(all_keywords).most_common(15)
        kw_df = pd.DataFrame(kw_counts, columns=["keyword", "count"])
        fig4 = px.bar(kw_df, x="count", y="keyword", orientation="h")
        fig4.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig4, use_container_width=True)

    with col_ent:
        st.subheader("Top Entities")
        all_entities = [e[0] for ents in df["entities_parsed"] for e in ents]
        ent_counts = Counter(all_entities).most_common(15)
        ent_df = pd.DataFrame(ent_counts, columns=["entity", "count"])
        fig5 = px.bar(ent_df, x="count", y="entity", orientation="h", color_discrete_sequence=["#e74c3c"])
        fig5.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 2 — ASK THE SYSTEM
# ══════════════════════════════════════════
elif page == "💬 Ask the System":
    st.title("💬 Ask the System")
    st.markdown("Ask any question about AI and the future of work. The system retrieves relevant YouTube comments and generates a grounded answer.")

    question = st.text_input(
        "Your question:",
        placeholder="e.g. What do people think about AI replacing software developers?"
    )

    col_strat, col_k = st.columns(2)
    with col_strat:
        strategy_choice = st.selectbox(
            "Retrieval strategy:",
            ["Auto (Query Analysis)", "Hybrid", "Semantic", "MMR (Diverse)", 
             "Negative Only", "Positive Only", "BM25 (Keyword)"],
            index=0
        )
    with col_k:
        k = st.slider("Number of comments to retrieve:", min_value=3, max_value=7, value=5)
        if k > 5:
            st.caption("Higher values include less relevant comments and may reduce answer quality.")

    if st.button("🔍 Search & Answer", type="primary") and question:
        # step 1: analyze the query before retrieval
        with st.spinner("Analyzing query..."):
            analysis_prompt = f"""Analyze this query and respond in JSON only:
    {{"intent": "one of: factual_qa, sentiment_inquiry, topic_exploration, entity_search, summarization",
    "entities_mentioned": ["list companies, people, AI tools mentioned or empty list"],
    "sentiment_requested": "one of: any, positive, negative, neutral",
    "reasoning": "one sentence"}}

    Query: {question}"""
        try:
            analysis_response = llm.invoke(analysis_prompt)
            text = analysis_response.content.strip()
            analysis = json.loads(text[text.find("{"):text.rfind("}")+1])
        except:
            analysis = {"intent": "factual_qa", "entities_mentioned": [],
                       "sentiment_requested": "any", "reasoning": "defaulting to general QA"}

        # map intent to retrieval strategy
        intent_to_strategy = {
            "entity_search":     "BM25 (Keyword)",
            "sentiment_inquiry": "Semantic",
            "topic_exploration": "Hybrid",
            "summarization":     "Hybrid",
            "factual_qa":        "Hybrid"
        }

        if strategy_choice == "Auto (Query Analysis)":
            strategy = intent_to_strategy.get(
                analysis.get("intent", "factual_qa"),
                "Hybrid"
            )
            auto_selected = True
        else:
            strategy = strategy_choice
            auto_selected = False

        # show query analysis visibly before retrieval
        with st.expander("🧠 Query Analysis", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Intent", analysis.get("intent", "factual_qa"))
            col_b.metric("Entities Found", len(analysis.get("entities_mentioned", [])))
            col_c.metric("Sentiment Filter", analysis.get("sentiment_requested", "any"))
            if analysis.get("entities_mentioned"):
                st.caption(f"Entities detected: {', '.join(analysis.get('entities_mentioned', []))}")
                st.caption(f"Reasoning: {analysis.get('reasoning', '')}")
            if auto_selected:
                st.info(f"Auto-selected strategy: **{strategy}** based on detected intent")
            else:
                st.info(f"Using manually selected strategy: **{strategy}**")

        # step 2: retrieve and generate

        with st.spinner("Retrieving context and generating answer..."):

            # retrieve based on strategy
            if strategy == "Hybrid":
                k = 3
                comment_docs, comment_meta, transcript_docs, _, summary_docs = hybrid_retrieval(question, k_each=k)
            elif strategy == "MMR (Diverse)":
                comment_docs, comment_meta = mmr_retrieval(question, k=k)
                transcript_docs, _ = semantic_retrieval(question, transcripts_col, k=3)
                summary_docs, _ = semantic_retrieval(question, summaries_col, k=2)
            elif strategy == "Negative Only":
                comment_docs, comment_meta = semantic_retrieval(
                    question, comments_col, k=k,
                    filter_dict={"sentiment": {"$eq": "negative"}}
                )
                transcript_docs, _ = semantic_retrieval(question, transcripts_col, k=3)
                summary_docs, _ = semantic_retrieval(question, summaries_col, k=2)
            elif strategy == "Positive Only":
                comment_docs, comment_meta = semantic_retrieval(
                    question, comments_col, k=k,
                    filter_dict={"sentiment": {"$eq": "positive"}}
                )
                transcript_docs, _ = semantic_retrieval(question, transcripts_col, k=3)
                summary_docs, _ = semantic_retrieval(question, summaries_col, k=2)
            elif strategy == "BM25 (Keyword)":
                # lexical retrieval - finds comments containing exact query terms
                import numpy as np
                scores = bm25_index.get_scores(question.lower().split())
                top_indices = np.argsort(scores)[::-1][:k]
                comment_docs = [bm25_texts[i] for i in top_indices]
                comment_meta = [{"video_name": df.iloc[i]["video_name"],
                                "sentiment": df.iloc[i]["sentiment"],
                                "topic_name": df.iloc[i]["topic_name"],
                                "likes": int(df.iloc[i]["likes"]),
                                "confidence": float(df.iloc[i]["confidence"])}
                                for i in top_indices]
                transcript_docs, _ = semantic_retrieval(question, transcripts_col, k=2)
                summary_docs, _ = semantic_retrieval(question, summaries_col, k=2)
            else:  # Semantic
                comment_docs, comment_meta = semantic_retrieval(question, comments_col, k=k)
                transcript_docs, _ = semantic_retrieval(question, transcripts_col, k=3)
                summary_docs, _ = semantic_retrieval(question, summaries_col, k=2)

            # generate answer
            with mlflow.start_run(run_name=f"dashboard_{question[:30]}"):
                mlflow.log_param("question", question)
                mlflow.log_param("strategy", strategy)
                mlflow.log_param("k", k)
                mlflow.log_param("auto_selected", auto_selected)
                mlflow.log_param("detected_intent", analysis.get("intent", "unknown"))
                mlflow.log_param("entities_found", str(analysis.get("entities_mentioned", [])))
    
                answer = generate_answer(question, comment_docs, transcript_docs, summary_docs)
    
                mlflow.log_metric("comments_retrieved", len(comment_docs))
                mlflow.log_text(answer, "answer.txt")
                
        # display answer
        st.markdown("### 🤖 Answer")
        st.info(answer)

        # display retrieved comments
        st.markdown("### 📝 Retrieved Comments")
        for i, (doc, meta) in enumerate(zip(comment_docs, comment_meta)):
            sentiment_color = {
                "negative": "🔴",
                "neutral": "⚪",
                "positive": "🟢"
            }.get(meta.get("sentiment", "neutral"), "⚪")

            with st.expander(f"{sentiment_color} Comment {i+1} — {meta.get('video_name', '')} | {meta.get('sentiment', '')}"):
                st.write(doc)
                col1, col2, col3 = st.columns(3)
                col1.metric("Likes", meta.get("likes", 0))
                col2.metric("Confidence", f"{meta.get('confidence', 0):.2f}")
                col3.metric("Topic", meta.get("topic_name", "")[:30])

# ══════════════════════════════════════════
# PAGE 3 — BIAS EXPLORER
# ══════════════════════════════════════════
elif page == "⚖️ Bias Explorer":
    st.title("⚖️ Bias Explorer")
    st.markdown("""
    This page demonstrates how retrieval bias affects generated answers.
    The same question, filtered to different sentiment groups, produces completely different conclusions.
    This mirrors how commercial LLMs reflect biases in their training data.
    """)

    st.info(f"**Dataset sentiment:** 50.2% negative | 31.2% neutral | 18.7% positive — caused by video selection bias toward negatively-framed titles.")

    question = st.text_input(
        "Enter a question to see bias in action:",
        value="What will happen to jobs because of AI?"
    )

    if st.button("⚖️ Compare Perspectives", type="primary") and question:

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI analyst. Answer based only on the provided comments. Be direct and concise."),
            ("human", "Question: {question}\n\nComments:\n{context}\n\nAnswer in 3-4 sentences.")
        ])

        with st.spinner("Generating answers from different perspectives..."):
            # negative perspective
            neg_docs, neg_meta = semantic_retrieval(
                question, comments_col, k=5,
                filter_dict={"sentiment": {"$eq": "negative"}}
            )
            neg_answer = (qa_prompt | llm).invoke({
                "question": question,
                "context": "\n\n".join(neg_docs)
            }).content

            # positive perspective
            pos_docs, pos_meta = semantic_retrieval(
                question, comments_col, k=5,
                filter_dict={"sentiment": {"$eq": "positive"}}
            )
            pos_answer = (qa_prompt | llm).invoke({
                "question": question,
                "context": "\n\n".join(pos_docs)
            }).content

            # neutral perspective
            neu_docs, neu_meta = semantic_retrieval(
                question, comments_col, k=5,
                filter_dict={"sentiment": {"$eq": "neutral"}}
            )
            neu_answer = (qa_prompt | llm).invoke({
                "question": question,
                "context": "\n\n".join(neu_docs)
            }).content

        # display side by side
        col_neg, col_neu, col_pos = st.columns(3)

        with col_neg:
            st.markdown("### 🔴 Negative Perspective")
            st.markdown(f"*Based on {len(neg_docs)} negative comments*")
            st.error(neg_answer)
            st.markdown("**Sample comments:**")
            for doc in neg_docs[:2]:
                st.caption(f"• {doc[:100]}...")

        with col_neu:
            st.markdown("### ⚪ Neutral Perspective")
            st.markdown(f"*Based on {len(neu_docs)} neutral comments*")
            st.warning(neu_answer)
            st.markdown("**Sample comments:**")
            for doc in neu_docs[:2]:
                st.caption(f"• {doc[:100]}...")

        with col_pos:
            st.markdown("### 🟢 Positive Perspective")
            st.markdown(f"*Based on {len(pos_docs)} positive comments*")
            st.success(pos_answer)
            st.markdown("**Sample comments:**")
            for doc in pos_docs[:2]:
                st.caption(f"• {doc[:100]}...")

        st.markdown("---")
        st.markdown("### 💡 Key Insight")
        st.markdown("""
        These three answers were generated by the **same system**, with the **same question**, 
        from the **same dataset** — the only difference is which comments were retrieved.
        
        This demonstrates that RAG systems are not neutral: they inherit and amplify the 
        biases present in their underlying data. In this dataset, 50.2% of comments are 
        negative — meaning default retrieval will disproportionately surface pessimistic views 
        about AI and jobs, regardless of what the question actually asks for.
        """)

# ══════════════════════════════════════════
# PAGE 4 — TOPIC & ENTITY EXPLORER
# ══════════════════════════════════════════
elif page == "🔍 Topic & Entity Explorer":
    st.title("🔍 Topic & Entity Explorer")

    tab1, tab2, tab3 = st.tabs(["📌 Topics", "🏷️ Entities", "🕸️ Knowledge Graph"])

    # ── TAB 1: TOPICS ──
    with tab1:
        st.subheader("Topic Distribution")

        topic_counts = df[df["topic"] != -1]["topic_name"].value_counts().head(20)
        fig = px.bar(
            x=topic_counts.values,
            y=topic_counts.index,
            orientation="h",
            color=topic_counts.values,
            color_continuous_scale="Blues",
            labels={"x": "Comment Count", "y": "Topic"}
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Explore a Topic")

        topic_options = df[df["topic"] != -1]["topic_name"].value_counts().index.tolist()
        selected_topic = st.selectbox("Select a topic:", topic_options)

        if selected_topic:
            topic_df = df[df["topic_name"] == selected_topic]

            col1, col2, col3 = st.columns(3)
            col1.metric("Comments", len(topic_df))
            col2.metric("Avg Likes", f"{topic_df['likes'].mean():.1f}")
            col3.metric("Dominant Sentiment",
                       topic_df["sentiment"].value_counts().index[0])

            # sentiment breakdown for this topic
            sent_counts = topic_df["sentiment"].value_counts()
            fig2 = px.pie(
                values=sent_counts.values,
                names=sent_counts.index,
                color=sent_counts.index,
                color_discrete_map={
                    "negative": "#e74c3c",
                    "neutral": "#95a5a6",
                    "positive": "#2ecc71"
                }
            )
            st.plotly_chart(fig2, use_container_width=True)

            # sample comments from this topic
            st.subheader("Sample Comments")
            samples = topic_df.sample(min(5, len(topic_df)))
            for _, row in samples.iterrows():
                sentiment_icon = {"negative": "🔴", "neutral": "⚪", "positive": "🟢"}.get(row["sentiment"], "⚪")
                st.markdown(f"{sentiment_icon} {row['text'][:200]}...")

    # ── TAB 2: ENTITIES ──
    with tab2:
        st.subheader("Entity Analysis")

        col_type, col_n = st.columns(2)
        with col_type:
            entity_type = st.selectbox(
                "Entity type:",
                ["All", "ORG", "PRODUCT", "PERSON", "CONCEPT", "GPE"]
            )
        with col_n:
            top_n = st.slider("Show top N entities:", 5, 30, 15)

        # filter by type
        all_entities = []
        for ents in df["entities_parsed"]:
            for text, label in ents:
                if entity_type == "All" or label == entity_type:
                    all_entities.append(text)

        ent_counts = Counter(all_entities).most_common(top_n)
        ent_df = pd.DataFrame(ent_counts, columns=["entity", "count"])

        color_map = {
            "ORG": "#e74c3c",
            "PRODUCT": "#3498db",
            "PERSON": "#f39c12",
            "CONCEPT": "#95a5a6",
            "GPE": "#2ecc71",
            "All": "#9b59b6"
        }

        fig3 = px.bar(
            ent_df, x="count", y="entity", orientation="h",
            color_discrete_sequence=[color_map.get(entity_type, "#9b59b6")]
        )
        fig3.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig3, use_container_width=True)

        # entity sentiment breakdown
        st.markdown("---")
        st.subheader("How is an entity discussed?")
        entity_search = st.text_input("Search for an entity:", placeholder="e.g. Google, ChatGPT, Sam Altman")

        if entity_search:
            mask = df["entities"].str.contains(entity_search, case=False, na=False)
            entity_df = df[mask]

            if len(entity_df) > 0:
                st.markdown(f"Found **{len(entity_df)}** comments mentioning **{entity_search}**")
                sent_breakdown = entity_df["sentiment"].value_counts(normalize=True) * 100
                col1, col2, col3 = st.columns(3)
                col1.metric("Negative", f"{sent_breakdown.get('negative', 0):.1f}%")
                col2.metric("Neutral", f"{sent_breakdown.get('neutral', 0):.1f}%")
                col3.metric("Positive", f"{sent_breakdown.get('positive', 0):.1f}%")

                st.subheader("Sample comments mentioning this entity:")
                for _, row in entity_df.sample(min(3, len(entity_df))).iterrows():
                    st.caption(f"• [{row['sentiment']}] {row['text'][:150]}...")
            else:
                st.warning(f"No comments found mentioning '{entity_search}'")

    # ── TAB 3: KNOWLEDGE GRAPH ──
    with tab3:
        st.subheader("Entity Relationship Knowledge Graph")
        st.markdown("Extracted from 476 entity-rich comments using dependency parsing (spaCy + GLiNER).")

        # display saved knowledge graph image
        kg_path = "data/processed/knowledge_graph.png"
        if os.path.exists(kg_path):
            st.image(kg_path, use_container_width=True)
        else:
            st.warning("Knowledge graph image not found. Run notebook 02c_knowledge_graph.ipynb first.")

        # show top connections
        st.markdown("---")
        st.subheader("Most Connected Entities")
        top_entities_data = {
            "AI": 457, "Google": 56, "US": 52, "LLM": 40,
            "ChatGPT": 32, "Gemini": 31, "Microsoft": 31,
            "Claude": 30, "YouTube": 30, "AGI": 29
        }
        fig4 = px.bar(
            x=list(top_entities_data.values()),
            y=list(top_entities_data.keys()),
            orientation="h",
            color=list(top_entities_data.values()),
            color_continuous_scale="Reds"
        )
        fig4.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig4, use_container_width=True)