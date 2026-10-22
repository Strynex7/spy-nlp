from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

reviews = [
    ("Great product, works perfectly and delivery was fast", 1),
    ("Excellent quality, totally worth the money", 1),
    ("Terrible quality, stopped working after two days", 0),
    ("Worst purchase ever, complete waste of money", 0),
    ("बहुत बढ़िया प्रोडक्ट है, पैसे वसूल", 1),
    ("क्वालिटी शानदार है और डिलीवरी भी तेज़ थी", 1),
    ("बहुत खराब प्रोडक्ट है, पैसे बर्बाद हो गए", 0),
    ("दो दिन में ही काम करना बंद कर दिया, बेकार क्वालिटी", 0),
    ("Excelente producto, funciona perfectamente y llegó rápido", 1),
    ("Muy buena calidad, vale totalmente la pena", 1),
    ("Producto terrible, dejó de funcionar en dos días", 0),
    ("La peor compra, una completa pérdida de dinero", 0),
    ("Excellent produit, il fonctionne parfaitement et la livraison était rapide", 1),
    ("Très bonne qualité, ça vaut vraiment le prix", 1),
    ("Produit horrible, il a cessé de fonctionner après deux jours", 0),
    ("Le pire achat, une perte d'argent totale", 0),
    ("Très satisfait de mon achat, je le recommande", 1),
    ("La livraison était en retard et la boîte était abîmée", 0),
    ("Très déçu, c'est un faux produit et inutile", 0),
]

texts = [t for t, y in reviews]
labels = [y for t, y in reviews]

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X = vectorizer.fit_transform(texts)
model = LogisticRegression(max_iter=1000).fit(X, labels)

senses = {
    "light": {
        "weight": "light lightweight easy to carry portable weight ligero léger हल्का ले जाना आसान",
        "lamp": "light lamp bulb bright dim dark room luz lumière बल्ब रोशनी",
    },
    "cheap": {
        "affordable": "cheap price affordable value for money budget barato pas cher सस्ता पैसे वसूल",
        "poor quality": "cheap flimsy fragile plastic broke poor quality frágil cassé bon marché",
    },
}

sense_score = {
    "weight": 0.3,
    "lamp": 0.0,
    "affordable": 0.3,
    "poor quality": -0.3,
}

words = {
    "light": ["light", "ligero", "léger", "हल्का"],
    "cheap": ["cheap", "barato", "bon marché", "सस्ता"],
}

def find_word(review):
    for key, forms in words.items():
        for form in forms:
            if form in review.lower():
                return key, form
    return None, None

def find_sense(review, key, form):
    context = vectorizer.transform([review.lower().replace(form, "")])
    best, best_sim = None, -1
    for sense, desc in senses[key].items():
        sim = cosine_similarity(context, vectorizer.transform([desc]))[0][0]
        if sim > best_sim:
            best, best_sim = sense, sim
    return best

def analyze(review):
    prob = model.predict_proba(vectorizer.transform([review]))[0]
    score = prob[1] - prob[0]
    key, form = find_word(review)
    sense = None
    if key:
        sense = find_sense(review, key, form)
        score += sense_score[sense]
    label = "Positive" if score > 0 else "Negative"
    return form, sense, round(score, 3), label

tests = [
    "The laptop is so light and easy to carry, love it",
    "The room light is dim, the bulb stopped working, very bad",
    "Cheap price and great value for money, very happy",
    "Feels cheap and flimsy, the plastic broke in a week",
    "यह सस्ता है और पैसे वसूल, बहुत खुश हूं",
    "El portátil es muy ligero y fácil de llevar, me encanta",
    "Se siente barato y frágil, se rompió en una semana, muy malo",
    "Le téléphone est léger et très pratique, excellent",
    "Great product, fast delivery, highly recommend",
    "Très mauvais, livraison en retard, déçu",
]

for review in tests:
    form, sense, score, label = analyze(review)
    print("Review    :", review)
    print("Ambiguous :", form, "->", sense)
    print("Sentiment :", label, score)
    print()
