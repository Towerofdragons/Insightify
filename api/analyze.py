"""
Perform Sentiment analysis and generate headings with a Hugging face model
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from db.db import get_session, Article
import os
import logging
from dotenv import load_dotenv
import glob

# Create a logger 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load the Sentiment analyzer from Huggingface
# Using cardiffnlp/twitter-roberta-base-sentiment or prosusai/finbert


load_dotenv()  # Loads .env file

DEBUG_MODE = os.getenv("DEBUG_MODE")

sentiment_model_name = "models--cardiffnlp--twitter-roberta-base-sentiment"

embedding_model_name = "models--sentence-transformers--all-MiniLM-L6-v2"
if DEBUG_MODE == "false":
  mounted_container_path = "/models/hub"
else:
  mounted_container_path = "C:\\Users\Z.BOOK\.cache\huggingface\hub"

# Resolve local paths
sentiment_model_path = os.path.join(mounted_container_path, sentiment_model_name)
embedding_model_path = os.path.join(mounted_container_path, embedding_model_name)


LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}


def resolve_snapshot_path(base_path: str):
    """Resolve the actual snapshot directory under the Hugging Face cache layout."""
    snapshot_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
    if snapshot_dirs:
        return snapshot_dirs[0]  # assume first snapshot
    
    print("\n\n No snapshots found")
    return base_path  # fallback if structure differs


def load_sentiment_pipeline(model_path: str):
    """
    Load sentiment models from volume
    """

    resolved_path = resolve_snapshot_path(model_path)

    print("❗ Sentiment Resolved path:", resolved_path)
    print("Contents:", os.listdir(resolved_path) if os.path.exists(resolved_path) else "Path does not exist")


    if not os.path.exists(resolved_path):
        logger.error(f"Sentiment model path not found: {model_path}")
        raise FileNotFoundError(
            f"Sentiment model not found at {resolved_path}. "
            f"Ensure it’s mounted correctly into the container."
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        model = AutoModelForSequenceClassification.from_pretrained(resolved_path)

        tokenizer.save_pretrained("models/sentiment_model")
        model.save_pretrained("models/sentiment_model")
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

    resolved_path = resolve_snapshot_path(model_path)

    print("❗ Embedder Resolved path:", resolved_path)
    print("Contents:", os.listdir(resolved_path) if os.path.exists(resolved_path) else "Path does not exist")

    if not os.path.exists(resolved_path):
        logger.error(f"Embedding model path not found: {resolved_path}")
        raise FileNotFoundError(
            f"Embedding model not found at {resolved_path}. "
            f"Ensure it’s mounted correctly into the container."
        )
    try:
        embedder = SentenceTransformer(resolved_path)
        return embedder
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

  print("Update Sentiments")
  with get_session() as session:
    articles = session.query(Article).filter(Article.sentiment==None).all()

    if not articles:
      print("\nNo articles currently in storage OR missing sentiments\n")
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
      print("\nNo articles currently in storage OR missing embedding\n")
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