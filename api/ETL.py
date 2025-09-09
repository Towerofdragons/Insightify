import requests
import ssl, feedparser
from newspaper import article
import lxml
import pandas as pd
from datetime import date, timedelta

from dotenv import load_dotenv
import os

from db import init_db

init_db() # Create Table if not already present


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

  rss = feedparser.parse("https://www.kenyanews.go.ke/feed/")

  print (rss)

  scraped = []
  for e in rss.entries[:5]:
      art = article(e.link)
      try:
          art.download(); art.parse(); art.nlp()
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
  df_rss = pd.DataFrame(scraped)

  print(df_rss)

# Combine and dedupe
#df = pd.concat([df_api, df_rss], ignore_index=True).drop_duplicates("url")

#print(f"Inserted {len(df)} articles")

def aggregate(concatenated_dataframe):
  """
  IDK what this does yet

  This puts the feeds together into a news stream
  """

  # Insert into Postgres
  engine = create_engine("postgresql+psycopg2://insightify:password@insightify-database:5432/insightify_db")
  df.to_sql("articles", engine, if_exists="append", index=False)
  #df = pd.concat([df_api, df_rss], ignore_index=True).drop_duplicates("url")
 


#get_API_articles()
#get_RSS_feeds()