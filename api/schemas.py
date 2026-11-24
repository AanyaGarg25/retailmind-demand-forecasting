from pydantic import BaseModel

class ForecastRequest(BaseModel):
    store: str
    item: str