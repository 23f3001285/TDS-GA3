import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Request Model
# -----------------------------
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

# -----------------------------
# Response Model
# -----------------------------
class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

# -----------------------------
# POST /comment Endpoint
# -----------------------------
@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(request: CommentRequest):

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="API key not configured")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis system. "
                               "Classify sentiment strictly as positive, negative, or neutral. "
                               "Return rating 1 (very negative) to 5 (very positive)."
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )

        return response.output_parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
