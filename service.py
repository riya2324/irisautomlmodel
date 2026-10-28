import bentoml
from bentoml.io import JSON

@bentoml.service
class IrisService:

    @bentoml.api
    def predict(self, input_data: dict) -> dict:

        return {
            "prediction": "setosa"
        }