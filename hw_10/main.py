import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import Query
from pydantic import BaseModel

from model.utils import read_model

app = FastAPI()

model_path = 'model/model.pkl'
model = read_model(model_path)


class ModelRequestData(BaseModel):
    total_square: float
    rooms: int
    floor: int
    lat: float
    lon: float
    city: str

class Result(BaseModel):
    result: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "I am alive!"}, status_code=200)


@app.post("/predict_post", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

@app.get("/predict_get", response_model=Result)
def preprocess_data(
    total_square: float = Query(...),
    rooms: int = Query(...),
    floor: int = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    city: str = Query(...)
):
    input_df = pd.DataFrame([{
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "lat": lat,
        "lon": lon,
        "city": city
    }])
    result = model.predict(input_df)[0]
    return Result(result=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)