"""
Perform Sentiment analysis and generate headings with a Hugging face model
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from db.db import get_session, Article
import os
import logging

# Create a logger 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load the Sentiment analyzer from Huggingface
# Using cardiffnlp/twitter-roberta-base-sentiment or prosusai/finbert

sentiment_model_name = "models--cardiffnlp-twitter-roberta-base-sentiment"

embedding_model_name = "models--sentence-transformers-all-MiniLM-L6-v2"

mounted_container_path = "/models/hub"

# Resolve local paths
sentiment_model_path = os.path.join(mounted_container_path, sentiment_model_name)
embedding_model_path = os.path.join(mounted_container_path, embedding_model_name)


LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}


def load_sentiment_pipeline(model_path: str):
    """
    Load sentiment models from volume
    """

    if not os.path.exists(model_path):
        logger.error(f"Sentiment model path not found: {model_path}")
        raise FileNotFoundError(
            f"Sentiment model not found at {model_path}. "
            f"Ensure it’s mounted correctly into the container."
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return pipeline("sentiment-analysis", 
                        model=model, 
                        tokenizer=tokenizer)
    except Exception as e:
        logger.exception("Failed to load sentiment analysis model")
        raise e

def load_embedder(model_path: str):
    """
    Load embedding models from volume
    """

    if not os.path.exists(model_path):
        logger.error(f"Embedding model path not found: {model_path}")
        raise FileNotFoundError(
            f"Embedding model not found at {model_path}. "
            f"Ensure it’s mounted correctly into the container."
        )
    try:
        return SentenceTransformer(model_path)
    except Exception as e:
        logger.exception("Failed to load embedding model")
        raise e


# Attempt to load the models from volumes
pipe = load_sentiment_pipeline(sentiment_model_path)
embedder = load_embedder(embedding_model_path)

logger.info("✅ Models loaded successfully.")


def perform_sentiment_analysis(text: str): # Limit anything going into the model to strings
  """
  Perform sentiment analysis on db records that do not have analyses installed.

  Expect return of (label, score) tuple
  """

  if not text:
    return (None, None)
  
  result = pipe(text)[0] # attempt providing all the content of an article

  return result["label"], float(result["score"]) # Return the label and the analysis score as a float

def get_embedding(text):
  """
  Calculate Embeddings for some provided content.
  """
  if not text:
    return None
  
  return embedder.encode(text).tolist()

def update_sentiments():
  """
  Fetch artcles without sentiment scores
  Process scores
  Write score to respective db row
  """

  with get_session() as session:
    articles = session.query(Article).filter(Article.sentiment==None).all()

    if not articles:
      print("No articles currently in storage OR missing sentiments")
      return None


    for article in articles:
      try:
        text = (article.title or " ") + " " + (article.description or " ") + " " + article.content if article.content != None else " "

        # Analyze Content

        label, score = perform_sentiment_analysis(text[:512])

        if label:
          print(f" Sentiment for {article. title} is {LABEL_MAP[label]}")

          article.sentiment = score

      except Exception as e:
        print(f"Error processing sentiment score: \n {e}")

      session.commit()


def get_embeddings():
  """
  Fetch artcles without embeddings
  Process embeddings
  Write embedding to respective db row
  """

  with get_session() as session:
    articles = session.query(Article).filter(Article.embedding==None).all()

    if not articles:
      print("No articles currently in storage OR missing embedding")
      return None


    for article in articles:
      try:
        text = (article.title or " ") + " " + (article.description or " ") + " " + article.content if article.content != None else " "

        # Analyze Content

        embedding = get_embedding(text)

        if embedding:
          print(f" Embedding for {article. title} calculated")

          article.embedding = embedding

      except Exception as e:
        print(f"Error processing embedding: \n {e}")

      session.commit()



if __name__ == "__main__":
  update_sentiments()
  get_embeddings()