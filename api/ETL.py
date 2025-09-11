import requests
import urllib3
import ssl, feedparser
from newspaper import Article
import lxml
import pandas as pd
from datetime import date, timedelta

from dotenv import load_dotenv
from newspaper.configuration import Configuration

import os

from db.db import init_db

#init_db() # Create Table if not already present

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

today = date.today()
yesterday =  today - timedelta(days=1)

day_before_yesterday = today - timedelta(days=2)

load_dotenv()


url = ('https://newsdata.io/api/1/latest?'
       f'apikey={os.getenv("API_KEY")}&'
       'country=ke&'
       'language=en&')


RSS_feeds = [
   "https://www.standardmedia.co.ke/rss/kenya.php",
   "https://www.kenyanews.go.ke/feed/",
   
]

def get_API_articles():

  """
  Get data from News Data API for kenyan feeds and store in Postgres
  """
  response = requests.get(url)

 # print(response.text)
  response = response.json()

  if response['status'] == 'success':

    if response['results']:

      for item in response['results']:

        print (item['title'])

    else:
      print("Didn't find anything here.")

  else:
    print(f"Something went wrong \n {response['code']}")


def get_RSS_feeds():
  """Source News Data from Available RSS feeds and store in postgres"""
  # Scrape specific Kenyan RSS feed (e.g., Daily Nation or Kenyanews)

  if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

  feed = feedparser.parse("https://www.kenyanews.go.ke/feed/")

  print (feed)

  scraped = []
  for e in feed.entries:
 
      art = Article(e.link, verify=False, # Disable SSL check a
                    request_timeout=30) # Expand Timeout 30 seconds

      try:
          # Download and parse the article

          # fetch manually with SSL verification disabled
          # resp = requests.get(e.link, verify=False, timeout=10)
          # resp.raise_for_status()

          art.download()
          art.parse(); #art.nlp()
          scraped.append({
              "title": art.title,
              "description": art.summary,
              "url": e.link,
              "source": "Daily Nation",
              "published_at": e.published,
              "country": "ke"
          })
      except Exception as exc:
          print("Failed to fetch:", e.link, exc)
  df_feed = pd.DataFrame(scraped)

  print(df_feed)

# Combine and dedupe
#df = pd.concat([df_api, df_rss], ignore_index=True).drop_duplicates("url")

#print(f"Inserted {len(df)} articles")

def normalize_rss(concatenated_dataframe):
  """
  IDK what this does yet

  This puts the feeds together into a news stream
  """

  df.to_sql("articles", engine, if_exists="append", index=False)
  #df = pd.concat([df_api, df_rss], ignore_index=True).drop_duplicates("url")
 
  pass


def normalize_NewsDataAPI():
   pass

def ingest():
  """
  Take in all normalized data and put in postgres
  """
  pass
   

def scrape():
   """
   Trying out Newspaper module proper!
   """

   pass



#get_API_articles()
get_RSS_feeds()

