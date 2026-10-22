import re
import random
import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("twitter_samples")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(tweet):
    tweet = re.sub(r"https?://\S+", "", tweet)
    tweet = re.sub(r"#", "", tweet)
    tokens = tokenizer.tokenize(tweet)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def features(tokens):
    return {word: True for word in tokens}

positive = twitter_samples.strings("positive_tweets.json")
negative = twitter_samples.strings("negative_tweets.json")

data = [(features(preprocess(t)), "positive") for t in positive]
data += [(features(preprocess(t)), "negative") for t in negative]
random.seed(42)
random.shuffle(data)

train_set = data[:8000]
test_set = data[8000:]

classifier = NaiveBayesClassifier.train(train_set)
print("Accuracy:", accuracy(classifier, test_set))
classifier.show_most_informative_features(10)

sia = SentimentIntensityAnalyzer()

tweets = [
    "I love this phone, it is amazing!",
    "This is the worst service ever",
    "The meeting is at 3 pm tomorrow",
    "Not bad, actually quite good",
    "I hate waiting in long queues",
    "Such a wonderful day with friends"
]

results = []
for tweet in tweets:
    tokens = preprocess(tweet)
    nb_label = classifier.classify(features(tokens))
    score = sia.polarity_scores(tweet)["compound"]
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    results.append((tweet, tokens, nb_label, label, score))

results.sort(key=lambda x: x[4], reverse=True)

for tweet, tokens, nb_label, label, score in results:
    print("Tweet     :", tweet)
    print("Tokens    :", tokens)
    print("NB Label  :", nb_label)
    print("VADER     :", label, round(score, 3))
    print()
