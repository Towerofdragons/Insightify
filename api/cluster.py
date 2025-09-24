"""
Perform HDBSCAN clustering on embeddings and create tags for clustered topics on the db
"""

from db.db import get_session

import numpy as np
import hdbscan

def cluster():
  """
  Perform HDBSCAN clustering and label
  """

  with get_session() as session:

    # Fetch unclustered articles
    articles = session.query(Article).filter(Article.cluster_id == None).all()

    ids = [a.id for a in articles]
    embeddings = [a.embedding for a in articles]
    titles = [a.title for a in articles]
    descriptions = [a.description for a in articles]


    if embeddings:
      x = np.array(embeddings)

      clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        metric="euclidian"
      )

    else:
      print("No unclustered embeddings found")
      return None


    labels = clusterer.fit_predict(x)
    probs = clusterer.probabilities_