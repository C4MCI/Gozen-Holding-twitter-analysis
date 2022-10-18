import os
from dotenv import load_dotenv
import tweepy as tw
import pandas as pd

load_dotenv()
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
auth = tw.OAuthHandler(API_KEY, SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tw.API(auth)
query = 'gozen holding OR gözen holding'
tweets = tw.Cursor(api.search_full_archive, query=query, label="productMindful").items(10000)
tweets_id = [tweet.id for tweet in tweets]


def get_text(status, twid):
    try:
        if "gözen holding" in status.full_text.lower() or "gozen holding" in status.full_text.lower():
            return twid, status.full_text
    except AttributeError:
        pass


text_col = []
id_col = []
for twid in tweets_id:
    status = api.get_status(twid, tweet_mode="extended")
    try:
        id, text = get_text(status, twid)
        id_col.append(id)
        text_col.append(text)
    except TypeError:
        pass
tweets = pd.DataFrame(text_col, index=id_col, columns=["text"])
tweets.to_csv("tweets.csv", encoding="utf-8")
