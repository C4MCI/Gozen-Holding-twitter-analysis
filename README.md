# Gozen-Holding-twitter-analysis
A sentiment analysis project about Gözen Holding.
I mostly used Twitter data for this project (see get_tweets.py). I also added some entries from Ekşi Sözlük by hand since I don't have access to their API service and there are only a few entries about Gözen Holding.

## Fetching Data From Twitter
In order to fetch data from Twitter, we need an API Key and Product Key. After getting those keys we can use this line of code.
```python
api = tw.API(auth)
query = 'gozen holding OR gözen holding'
tweets = tw.Cursor(api.search_full_archive, query=query, label="productMindful").items(10000)
```
This returns all tweets including the words gözen and holding. We have to do some more work since there may be irrelevant tweets including both of these words. See get_tweets.py for further reference.

## Preprocessing the Data
We will apply natural language processing methods to our data in order to get more accurate results. Some of these methods are:
* Tokenization
* Spelling Correction
* Removing Stopwords
* Lemmatization
```python
data["text"] = data["text"].apply(lambda x: url_pattern.sub(r'', x))
data['text'] = data['text'].str.replace('\d+', '')
data["text"] = data["text"].apply(lambda x: nltk.word_tokenize(x))
data["text"] = data["text"].apply(lambda x: [spell.correction(word) for word in x])
data["text"] = data["text"].apply(lambda x: [word for word in x if word not in stop_words])
data["text"] = data["text"].apply(lambda x: [word for word in x if isinstance(word, str)])
data["text"] = data["text"].apply(lambda x: tokenizer.tokenize(' '.join(x)))
data["text"] = data["text"].apply(lemmatize)
data["text"] = data["text"].apply(lambda x: ' '.join(x))
```

## Visualizing the Data
We processed our data. That means we can use some graphs to get information about it.

### Most used words along with Gözen Holding
![](https://i.imgur.com/w3fBJFP.png)
We can see that our dataset contains tweets in both Turkish and English. We will handle that later. Other than that everything looks like how it should be.

### Most used word combinations along with Gözen Holding (Bigram)
![](https://i.imgur.com/vyNY5if.png)

### Most used word combinations along with Gözen Holding (Trigram)
![](https://i.imgur.com/R38z0PY.png)


## Using Machine Learning to Determine Positivity
Since we have data in both Turkish and English, I will be using two different machine learning models. But first, we have to split our dataset into Turkish and English.

```python
def english_turkish_split(data):
    data["language"] = data["text"].apply(lambda x: langid.classify(x)[0])
    english_data = data[data["language"] == "en"]
    turkish_data = data[data["language"] == "tr"]
    return english_data, turkish_data
```
After that, I will use my pre-trained models to determine which tweets are positive or negative. To learn how I trained those models see [this repo](https://github.com/C4MCI/Turkish-twitter-politician-analysis).

```python
data = pd.read_csv("processed_tweets.csv", encoding="utf-8", index_col="id")
en_data = data[data["language"] == "en"]
tr_data = data[data["language"] == "tr"]

en_data = classify_english(en_data)
tr_data = classify_turkish(tr_data)

data = pd.concat([en_data, tr_data])
```

## Result
![](blob:https://imgur.com/3ccb0136-115b-4e67-86db-a73d7136b4b8)


Since tweets about Gözen Holding mostly consist of news, we can see that most of them are labeled as 'Notr'. Other than that, the company seems to have a great public look.




