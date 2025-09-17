"""
Perform Sentiment analysis with a Hugging face model
"""

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from db.db import get_session, Article


# Load the Sentiment analyzer from Huggingface
# Using cardiffnlp/twitter-roberta-base-sentiment or prosusai/finbert

pipe = pipeline(
  "sentiment-analysis",
  model="cardiffnlp/twitter-roberta-base-sentiment"
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}


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