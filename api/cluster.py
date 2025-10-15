"""
Perform HDBSCAN clustering on embeddings and create tags for clustered topics on the db
"""

from db.db import get_session, Article, Cluster
import logging

import numpy as np
import hdbscan
from umap import UMAP

import re
import math
from collections import defaultdict
from typing import List, Dict


from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# Create a logger 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#TODO-Create main shared logger object and change messages to logs

# -----------------------------
# Helper methods
# -----------------------------
def sentence_splitter(text: str) -> List[str]:
    """
    Split Sentences on Punctuation -- No dependencies needed
    """
    if not text:
        return []
    # Replace newlines with spaces, collapse spaces
    text = re.sub(r'\s+', ' ', text.strip())
    # Split on sentence-ending punctuation (keeps punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def make_cluster_label(titles: List[str], max_words: int = 3) -> str:
    """
    Make a crude label from cluster titles by taking the most common words.    
    """
    words = []
    for t in titles:
        if not t:
            continue
        # simple tokenization
        tokens = re.findall(r'\w+', t.lower())
        words.extend([w for w in tokens if len(w) > 3])  # drop very short tokens

    if not words:
        return "Miscellaneous Cluster"

    # frequency
    freq = defaultdict(int)
    for w in words:
        freq[w] += 1
    sorted_words = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    chosen = [w for w, _ in sorted_words[:max_words]]
    return " ".join(chosen)

def build_cluster_prompt(titles: List[str], descriptions: List[str], top_n: int = 5, per_item_max_chars: int = 400) -> str:
    """
    Build the text chunk that will be fed to the summarizer for a cluster.
    Picks top_n articles and for each article combine title + description, shortended to
    per_item_max_chars. 
    Keeps the summarizer input within model limits.
    """
    items = []
    for t, d in zip(titles[:top_n], descriptions[:top_n]):
        text = (t or "") + ". " + (d or "")
        text = text.strip()
        if len(text) > per_item_max_chars:
            text = text[:per_item_max_chars].rsplit(" ", 1)[0] + "..."
        items.append(text)
    prompt = "\n\n".join(items)
    return prompt if prompt else "No content available."

# -----------------------------
# Summarization: initialize HF pipeline
# -----------------------------
def init_summarizer(model_name: str = "facebook/bart-large-cnn", device: int = -1):
    """
    Initialize the HF summarization pipeline. If you have GPU set device=0.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
    return summarizer

def summarize_cluster_text(summarizer, text: str, max_length: int = 130, min_length: int = 40) -> str:
    """
    Call the summarizer pipeline on the text and postprocess to produce
    a 3-line summary: we split into sentences and join the first 3 (or fewer).
    """
    if not text or len(text.strip()) == 0:
        return ""

    # HF summarizer prefers not-too-long inputs; assume we pre-truncated the prompt
    output = summarizer(text, max_length=max_length, min_length=min_length, truncation=True)
    summary_text = output[0]["summary_text"].strip()

    # Normalize and split into sentences
    sentences = sentence_splitter(summary_text)
    
    top3 = sentences[:3]
    # Ensures each 'line' is not too long; rejoins with newline chars for "3-line" effect
    return "\n".join(top3)




# -----------------------------
# Main method
# -----------------------------

def cluster():
  """
  Perform HDBSCAN clustering and label
  """

  unclustered_articles = None

  with get_session() as session:

    # Fetch unclustered articles
    unclustered_articles = session.query(Article).filter(Article.cluster_id == None).all()

    if not unclustered_articles:
      print("No unclustered articles. Exiting.")
      return

    ids = [a.id for a in unclustered_articles]
    embeddings = [a.embedding for a in unclustered_articles] 
    titles = [a.title for a in unclustered_articles]
    descriptions = [a.description for a in unclustered_articles]

    
    X = np.array(embeddings) # convert embeddings into numoy array

    # reduce dimensionality with UMAP for speed and noise reduction
    reducer = UMAP(n_components=50, random_state=42)
    X_reduced = reducer.fit_transform(X)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
      min_cluster_size=5,
      min_samples=2,
      metric="euclidean"
    )      

    labels = clusterer.fit_predict(X_reduced)
    probs = clusterer.probabilities_

    # Group indices by HDBSCAN label (ignore -1 / noise for creating clusters)
    clusters_map = defaultdict(list)
    for i, lbl in enumerate(labels):
        if lbl == -1:
            continue
        clusters_map[int(lbl)].append(i)

    if not clusters_map:
        print("No clusters found (all points considered noise).")
        return
    
    # Create Cluster rows and map HDBSCAN label -> DB cluster id
    cluster_id_map: Dict[int, int] = {}
    for lbl, indices in clusters_map.items():
        # Create a rough human-readable label from titles
        label_text = make_cluster_label([titles[i] for i in indices], max_words=3)
        cluster_row = Cluster(label=label_text, summary=None)
        session.add(cluster_row)
        session.flush()  # ensures cluster_row.id is assigned
        cluster_id_map[lbl] = cluster_row.id

    session.commit()  

  #TODO - SET UP SUMMARIZER model
    # NOTE: Init summarizer once (device=-1 for CPU; set 0 for first GPU)
    summarizer = init_summarizer(model_name="facebook/bart-large-cnn", device=-1)


     # For each new cluster pick top articles, build prompt, summarize, and update cluster.summary
    for lbl, indices in clusters_map.items():
        # Choose top-k articles to summarize. Use the clusterer's internal density measure
        # or simply take the first N. Here we'll sort by cluster probability to get representative items.
        idx_and_conf = [(i, probs[i]) for i in indices]
        # sort descending by confidence so representative points are first
        idx_and_conf_sorted = sorted(idx_and_conf, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in idx_and_conf_sorted[:6]]  # choose up to 6 articles

        # Build summarization prompt from those top articles
        cluster_titles = [titles[i] for i in top_indices]
        cluster_descriptions = [descriptions[i] for i in top_indices]
        prompt_text = build_cluster_prompt(cluster_titles, cluster_descriptions, top_n=6, per_item_max_chars=400)

        # Summarize and enforce 3-line output
        try:
            summary_three_lines = summarize_cluster_text(summarizer, prompt_text, max_length=120, min_length=30)
        except Exception as e:
            # If summarization fails for any reason, fallback to a simple auto-label + article titles
            summary_three_lines = " ".join(cluster_titles[:3])
            print(f"Summarization failed for cluster {lbl}: {e}")

        # Update cluster.summary in DB
        db_cluster_id = cluster_id_map[lbl]
        session.query(Cluster).filter(Cluster.id == db_cluster_id).update({
            Cluster.summary: summary_three_lines
        })
        session.commit()

      # Update article rows with cluster id + confidence
        for i in indices:
            article_db_id = ids[i]
            lbl_conf = float(probs[i])
            session.query(Article).filter(Article.id == article_db_id).update({
                Article.cluster_id: db_cluster_id,
                Article.cluster_confidence: lbl_conf
            })
        session.commit()

    print(f"Created and summarized {len(clusters_map)} clusters.")


if __name__ == "__main__":
    cluster()