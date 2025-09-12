import requests
import urllib3
import ssl, feedparser
from newspaper import Article
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
import os

from db.db import init_db, get_session, Source, Article


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
]


def get_NewsData_Sources():
  """
  Get News from NewsData API
  """

  response = requests.get(url)

 # print(response.text)
  response = response.json()

  if response.get('status') == 'success' and response.get('results', None):

    for item in response['results']:

      print (item['title'])

    else:
      print("Didn't find anything here.")

  else:
    print(f"Something went wrong \n {response['code']}")




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
           art = Article(e.link, verify=False, request_timeout=30)

           art.download()
           art.parse()

           all_articles.append(
              {
                 "title": art.title,
                 "description": art.text[:300], # First 300 chars
                  "url": e.link,
                  "source": feed.feed.get("title","RSS"),
                  "published_at": getattr(e, "published", None),
                  "country": "ke"
              }
           )
        except Exception as ex:
           print("Failed to Fetch RSS entry:", e.link, ex)

  print(all_articles)
  return(all_articles)

"""
Transform into DB schema
"""

def normalize_article(raw, source_type="api"):
    """
    Convert raw API/RSS dict into DB-ready dict
    """
    if source_type == "api":  # NewsData.io format
        return {
            "title": raw.get("title"),
            "description": raw.get("description"),
            "url": raw.get("link"),
            "author": raw.get("creator", None),
            "published_at": raw.get("pubDate"),
            "country": raw.get("country", ["ke"])[0] if raw.get("country") else "ke",
            "language": raw.get("language"),
            "raw": raw,  # store full payload
            "source_name": "NewsData"
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


"""
Load in DB
"""

def insert_articles(articles):
   with get_session as session:
        try:
          for article in articles:
          
            # Check if source exists, if not, add a new source and assign ID
            source = session.query(Source).filter_by(
                      name=article.get("source_name")).first()

            if not source:
              source = Source(name=article["source_name"])
              session.add(source)
              session.flush()  # assign ID


            existing_article = session.query(Article).filter_by(
                                    url=article.get("url")).first()
            
            #check if current article's url is already in db, else, move to next
            if existing_article:
              print("This is an existing article")
              continue
          
          
            new_Artcle = Article(  
                title=article["title"],
                description=article["description"],
                url=article["url"],
                author=article.get("author"),
                published_at=article.get("published_at"),
                country=article.get("country"),
                language=article.get("language"),
                raw=article["raw"],
                source=source

            )

            session.add(new_Artcle)

          session.commit()
        except Exception as e:
          print("Insertion failed with this exception", e)


"""
Combine above methods to ingest data
"""

def ingest():
  #Extract data from available sources
  API_raw = get_NewsData_Sources()
  RSS_raw = get_RSS_articles()

  #Transform raw into db storable

  API_normalized = [normalize_article(i, "api") for i in API_raw]
  RSS_normalized = [normalize_article(i, "rss") for i in RSS_raw]

  # COMBINE AND DEDUPLICATE

  all_articles = {i["url"]: i for i in (API_normalized + RSS_normalized)}.values()

  # Load all articles

  insert_articles(all_articles)
  print(f"Inserted {len(all_articles)} articles.")


if __name__ == "__main__":
  ingest()
