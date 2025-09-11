import requests
import urllib3
import ssl, feedparser
from newspaper import Article
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
import os

from db.db import init_db


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