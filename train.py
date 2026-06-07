import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    'email': [
        'Win money now',
        'Claim your free prize',
        'Meeting at 10 AM tomorrow',
        'Project submission deadline',
        'You won a lottery',
        'Lunch with team today'
    ],
    'label': [
        'spam',
        'spam',
        'ham',
        'ham',
        'spam',
        'ham'
    ]
}

df = pd.DataFrame(data)

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'])

# Train model
model = MultinomialNB()
model.fit(X, df['label'])

# Test email
test_email = ["Congratulations! You won a free iPhone"]
test_features = vectorizer.transform(test_email)

prediction = model.predict(test_features)

print("Prediction:", prediction[0])
