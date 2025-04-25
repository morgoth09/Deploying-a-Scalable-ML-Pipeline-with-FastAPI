import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics

# Sample data for testing
df = pd.read_csv("./data/census.csv")
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)



# TODO: implement the first test. Change the function name and input as needed
def test_model_type():
    """
    Test if the model returned is a RandomForestClassifier.
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_compute_metrics_range():
    """
    Test that precision, recall, and fbeta are between 0 and 1.
    """
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


# TODO: implement the third test. Change the function name and input as needed
def test_train_test_shapes():
    """
    Test that the train and test sets have matching feature dimensions.
    """
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
