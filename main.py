import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

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

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Analyze sentiment of this comment: {request.comment}",
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
