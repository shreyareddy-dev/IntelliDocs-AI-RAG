from typing import Optional

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    document: Optional[str] = None


class Source(BaseModel):
    document: str
    page: int


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]