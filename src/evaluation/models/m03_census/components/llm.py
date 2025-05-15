import os
import openai
import dashscope
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
        """Generate text using the LLM.
        
        Args:
            prompt: The input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated text.
        """
        pass

class OpenAILLM(BaseLLM):
    """OpenAI LLM wrapper."""
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize OpenAI client."""
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
        """Generate text using OpenAI API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"ERROR: OpenAI API call failed: {str(e)}")
            return ""

class QwenLLM(BaseLLM):
    """Qwen LLM wrapper using dashscope."""
    
    def __init__(self, model: str = "qwen-max"):
        """Initialize Qwen client."""
        self.model = model
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is required")
        dashscope.api_key = self.api_key
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
        """Generate text using Qwen through dashscope API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format='message'  # Get response in message format
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                print(f"ERROR: Qwen API call failed with status {response.status_code}")
                return ""
        except Exception as e:
            print(f"ERROR: Qwen API call failed: {str(e)}")
            return ""

def create_llm(provider: str = "openai", model: str = None) -> BaseLLM:
    """Factory function to create LLM instances.
    
    Args:
        provider: The LLM provider ("openai" or "qwen")
        model: Optional model name. If not provided, uses default for provider.
        
    Returns:
        An LLM instance.
    """
    if provider == "openai":
        return OpenAILLM(model=model if model else "gpt-4")
    elif provider == "qwen":
        return QwenLLM(model=model if model else "qwen-max")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 