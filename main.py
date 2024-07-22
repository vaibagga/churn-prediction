import redis
from fastapi import FastAPI

r = redis.Redis(host='localhost', port=6379, db=0)
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/{customer_id}")
def predict(customer_id):
    try:
        prediction = float(r.get(customer_id))
        label = "Yes" if prediction > 0.5 else "No"
        return {
            "status": 200,
            "customer_id": customer_id,
            "predicted_label": label,
            "churn_probability": prediction,
            "predicted_label_value": round(prediction)
        }
    except TypeError:
        return {
            "status": 404,
            "message": f"Id {customer_id} not found in records"
        }
