import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Create Sample Dataset
# -----------------------------
data = {
    'attention_level': np.random.randint(1,10,200),
    'quiz_score': np.random.randint(40,100,200),
    'time_spent': np.random.randint(10,120,200),
    'interaction_count': np.random.randint(1,20,200),
    'engagement': np.random.randint(0,2,200)
}

df = pd.DataFrame(data)

X = df.drop("engagement", axis=1)
y = df["engagement"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Deep Learning Model Function
# -----------------------------
def create_model():

    model = Sequential()

    model.add(Dense(16, activation='relu', input_shape=(4,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# -----------------------------
# Bagging Ensemble Training
# -----------------------------
n_models = 5
models = []

for i in range(n_models):

    X_sample, y_sample = resample(
        X_train, y_train,
        replace=True
    )

    model = create_model()

    model.fit(
        X_sample,
        y_sample,
        epochs=20,
        verbose=0
    )

    models.append(model)

# -----------------------------
# Ensemble Prediction
# -----------------------------
predictions = []

for model in models:
    pred = model.predict(X_test)
    predictions.append(pred)

predictions = np.array(predictions)

avg_prediction = np.mean(predictions, axis=0)

final_prediction = (avg_prediction > 0.5).astype(int)

accuracy = accuracy_score(y_test, final_prediction)

print("Bagging Ensemble Accuracy:", accuracy)