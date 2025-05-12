import os
from openai import AsyncOpenAI
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

class OpenAILLM:
    """Simple OpenAI LLM wrapper"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI client"""
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = 0.7) -> str:
        """
        Generate text using OpenAI API
        
        Args:
            prompt: Input prompt
            max_tokens: Optional maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") 