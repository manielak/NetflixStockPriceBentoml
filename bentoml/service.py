import bentoml
from typing import Dict, Any
from pydantic import BaseModel
from bentoml.io import JSON
import pandas as pd

model_runner = bentoml.picklable_model.get("netflixmodel:latest").to_runner()

service = bentoml.Service("netflixmodel", runners=[model_runner])


class ModelFeatures(BaseModel):
    date: str

@service.api(input=JSON(pydantic_model=ModelFeatures), output=JSON())

def classify(input_value: ModelFeatures) -> Dict[str, Any]:
    input_series = pd.DataFrame([input_value.date], columns=["ds"])
    result = model_runner.run(input_series)

    print(result)

    result['ds'] = result['ds'].astype(str)
    return result
