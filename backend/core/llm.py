import asyncio
from typing import Any, Dict, Protocol, Type, TypeVar
from pydantic import BaseModel
import instructor
import google.generativeai as genai
from backend.config import settings

T = TypeVar("T", bound=BaseModel)

class LLMConnector(Protocol):
    """Generic interface for interacting with LLM services."""
    async def connect(self) -> None:
        pass
    async def disconnect(self) -> None:
        pass
    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        pass
    async def structured_generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        response_model: Type[T],
        model: str,
        **kwargs: Any
    ) -> T:
        pass

class InstructorGeminiConnector(LLMConnector):
    """Implementation of LLMConnector using instructor and Gemini."""
    
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name="models/gemini-2.5-pro"),
            mode=instructor.Mode.GEMINI_JSON,
        )
        
    async def connect(self) -> None:
        pass
        
    async def disconnect(self) -> None:
        pass

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        # Default text generation without instructor schema
        raise NotImplementedError("Use structured_generate for InstructorGeminiConnector")

    async def structured_generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        response_model: Type[T],
        model: str = settings.model_haiku,
        **kwargs: Any
    ) -> T:
        """Generate structured output validated against a Pydantic model."""
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        return await asyncio.to_thread(
            self.client.messages.create,
            messages=messages,
            response_model=response_model,
            **kwargs,
        )

llm_client = InstructorGeminiConnector()