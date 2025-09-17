import requests
import urllib3
import ssl, feedparser
from newspaper import Article as NewspaperArticle
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
import os

from db.db import init_db, get_session, Source, Article
from analyze import get_embedding

# Surpress SSL Warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

today = date.today()
yesterday = today - timedelta(days=1)
day_before_yesterday = today - timedelta(days=2)

load_dotenv()

init_db()

"""
 EXTRACT NEWS SOURCES
"""

url = ('https://newsdata.io/api/1/latest?'
       f'apikey={os.getenv("API_KEY")}&'
       'country=ke&'
       'language=en&')

RSS_FEEDS = [
   "https://www.standardmedia.co.ke/rss/kenya.php",
   "https://www.kenyanews.go.ke/feed/",
   "https://www.kenyans.co.ke/feeds/news?_wrapper_format=html",
   "https://www.capitalfm.co.ke/news/feed/",
]


def get_NewsData_Sources():
  """
  Get News from NewsData API

  Use NewsAPI to get news content
  """

  response = requests.get(url)
  response = response.json()

  if response.get('status') == 'success' and response.get('results', None):
    return response['results']

  else:
    print(f"Something went wrong \n {response['code']}")
    return None




def get_RSS_articles():
  """
    Scrape  Articles from RSS feeds
  """

  if hasattr(ssl, "_create_unverified_context"):
        ssl._create_default_https_context = ssl._create_unverified_context


  all_articles = []

  for feed_url in RSS_FEEDS:
     feed= feedparser.parse(feed_url)

     for e in feed.entries:
        try:
           all_articles.append({
              "title": e.get("title"),
              "description": e.get("summary"),        # or .get("description")
              "url": e.get("link"),
              "source": feed.feed.get("title", "RSS"),
              "published_at": getattr(e, "published", None),
              "country": "ke"
          })

        except Exception as ex:
           print("Failed to Fetch RSS entry:", e.link, ex)

  return(all_articles)

"""
Transform into DB schema
"""

def normalize_article(raw, source_type="api"):
    """
    Convert raw API/RSS dict into DB-ready dict
    """

    # Add content fields
    if source_type == "api":  # NewsData.io format
        
        return {
            "title": raw.get("title"),
            "description": raw.get("description"),
            "url": raw.get("link"),
            "author": raw.get("creator", None),
            "published_at": raw.get("pubDate"),
            "country":  "ke",
            "language": 'en',
            "raw": raw,  # store full payload
            "source_name": raw.get("source_name", "NewsData")
        }
    elif source_type == "rss":
        return {
            "title": raw.get("title"),
            "description": raw.get("description"),
            "url": raw.get("url"),
            "author": None,
            "published_at": raw.get("published_at"),
            "country": raw.get("country"),
            "language": "en",
            "raw": raw,
            "source_name": raw.get("source", "RSS")
        }


def insert_articles(articles):
   """
    Load soley article data in DB
    Data available in DB later for Postprocessing
    """
   
   with get_session() as session:
        try:
          for article in articles:
          
            # Check if source exists, if not, add a new source and assign ID
            source = session.query(Source).filter_by(
                      name=article.get("source_name")).first()

            if not source:
              source = Source(name=article["source_name"])
              session.add(source)
              session.flush()  # assign new source ID


            existing_article = session.query(Article).filter_by(
                                    url=article.get("url")).first()
            
            #check if current article's url is already in db, else, move to next
            if existing_article:
              print("This is an existing article. Skipping.")
              continue
          
          
            content = fetch_full_context(article["url"])

            if content == None:
               content = None


            text = (article["title"] or "") + " " + (article["description"] or "") + " " + (content or "")

            new_Artcle = Article(  
                title=article["title"],
                description=article["description"],
                url=article["url"],
                author=article.get("author"),
                published_at=article.get("published_at"),
                country=article.get("country"),
                language=article.get("language"),
                content = content,
                raw=article["raw"],
                source=source
            )

            session.add(new_Artcle)

          session.commit()
        except Exception as e:
          print("Insertion failed with this exception", e)


def fetch_full_context(url):
  """
  Use Newspaper4K to fetch full articles for each db entry
  """

  try:
    article = NewspaperArticle(url, request_timeout=20)
    article.download()
    article.parse()

    return article.text
  
  except Exception as e:

    print(f"Unable to fetch full article for link: {url} \n Error: {e}")
    return None




def ingest():
  """
  Combine above methods to ingest data.
  """

  #Extract data from available sources
  API_raw = get_NewsData_Sources()
  if API_raw == None:
     print("No api Articles returned")
     return
  
  RSS_raw = get_RSS_articles()
  if RSS_raw == None:
     print("No RSS Articles returned")
     return
  
  #Transform raw into db storable
  API_normalized = [normalize_article(i, "api") for i in API_raw]
  RSS_normalized = [normalize_article(i, "rss") for i in RSS_raw]

  # COMBINE AND DEDUPLICATE

  all_articles = {i["url"]: i for i in (API_normalized + RSS_normalized)}.values()

  # Load all articles

  insert_articles(all_articles)
  print(f"Collected {len(all_articles)} articles.")




if __name__ == "__main__":
  ingest()
