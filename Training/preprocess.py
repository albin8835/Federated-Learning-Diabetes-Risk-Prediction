import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from joblib import dump

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Remove duplicates
df = df.drop_duplicates()

# Encode categorical columns
df["gender"] = df["gender"].map({
    "Male": 0,
    "Female": 1,
    "Other": 2
})

df["smoking_history"] = df["smoking_history"].map({
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5
})

# Features and Target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dump(scaler, "scaler.pkl")

# Save test data
pd.DataFrame(X_test).to_csv(
    "X_test.csv",
    index=False
)

pd.DataFrame(y_test).to_csv(
    "y_test.csv",
    index=False
)

# Shuffle
X_train, y_train = shuffle(
    X_train,
    y_train,
    random_state=42
)

# Split among clients
num_clients = 4

chunk_size = len(X_train) // num_clients

for i in range(num_clients):

    start = i * chunk_size

    end = (
        (i + 1) * chunk_size
        if i < num_clients - 1
        else len(X_train)
    )

    pd.DataFrame(
        X_train[start:end]
    ).to_csv(
        f"client_{i+1}_X.csv",
        index=False
    )

    pd.DataFrame(
        y_train.iloc[start:end]
    ).to_csv(
        f"client_{i+1}_y.csv",
        index=False
    )

print("\nPreprocessing Completed Successfully")
print(f"Total Samples: {len(df)}")
print(f"Features: {X.shape[1]}")