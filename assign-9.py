import re
import nltk
from nltk.corpus import wordnet as wn, stopwords

nltk.download("wordnet")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def tokens(text):
    words = re.findall(r"[a-z]+", text.lower())
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return [wn.morphy(w) or w for w in words]

def get_context(sentence, target):
    return [w for w in tokens(sentence) if w != target]

def sense_bag(synset):
    related = [synset]
    related += synset.hypernyms()
    related += synset.hyponyms()
    related += synset.part_meronyms()
    related += synset.member_meronyms()
    related += synset.part_holonyms()
    related += synset.member_holonyms()
    bag = set()
    for s in related:
        for lemma in s.lemma_names():
            bag.add(lemma.lower())
        bag.update(tokens(s.definition()))
    return bag

def walker(sentence, target, pos=wn.NOUN):
    context = get_context(sentence, target)
    scores = {}
    for synset in wn.synsets(target, pos=pos):
        bag = sense_bag(synset)
        matched = sorted(set(w for w in context if w in bag))
        scores[synset] = (len(matched), matched)
    best = max(scores, key=lambda s: scores[s][0])
    return best, scores

tests = [
    ("He went to the bank to deposit money and open an account", "bank"),
    ("The fisherman sat on the bank of the river near the water", "bank"),
    ("The bat flew out of the cave at night", "bat"),
    ("He hit the ball with a wooden bat", "bat"),
    ("The plant needs sunlight and water to grow leaves", "plant"),
    ("The plant employs workers to manufacture cars in the factory", "plant"),
]

for sentence, target in tests:
    best, scores = walker(sentence, target)
    print("Sentence :", sentence)
    print("Context  :", get_context(sentence, target))
    for s, (score, matched) in scores.items():
        print("  ", s.name(), "| score =", score, "|", matched)
    print("Best sense:", best.name(), "-", best.definition())
    print()
