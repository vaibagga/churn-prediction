import redis
from fastapi import FastAPI

r = redis.Redis(host='localhost', port=6379, db=0)
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/{customer_id}")
def predict(customer_id):
    prediction = float(r.get(customer_id))
    label = "Yes" if prediction > 0.5 else "No"
    return {
        "customer_id": customer_id,
        "predicted_label": label,
        "churn_probabiliry": prediction,
        "predicted_label_value": round(prediction)
    }
