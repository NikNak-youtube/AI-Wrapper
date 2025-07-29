import openai
import ollama
import json
import anthropic
import google.generativeai as genai
import groq
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None


@dataclass
class Message:
    """Standardized message format."""
    role: str  # "user", "assistant", "system"
    content: str


class LLMWrapper:
    """
    Unified wrapper for multiple LLM providers.
    
    Supports: OpenAI, Anthropic, Google Gemini, Groq, and Ollama
    """
    
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all available clients."""
        try:
            self.clients[LLMProvider.OPENAI] = openai.OpenAI()
        except Exception as e:
            print(f"Warning: OpenAI client initialization failed: {e}")
        
        try:
            self.clients[LLMProvider.ANTHROPIC] = anthropic.Anthropic()
        except Exception as e:
            print(f"Warning: Anthropic client initialization failed: {e}")
        
        try:
            # Google Gemini client is initialized when needed
            self.clients[LLMProvider.GOOGLE] = genai
        except Exception as e:
            print(f"Warning: Google client initialization failed: {e}")
        
        try:
            self.clients[LLMProvider.GROQ] = groq.Groq()
        except Exception as e:
            print(f"Warning: Groq client initialization failed: {e}")
        
        try:
            # Ollama client is initialized when needed
            self.clients[LLMProvider.OLLAMA] = ollama
        except Exception as e:
            print(f"Warning: Ollama client initialization failed: {e}")
    
    def set_api_key(self, provider: LLMProvider, api_key: str):
        """Set API key for a specific provider."""
        if provider == LLMProvider.OPENAI:
            self.clients[provider] = openai.OpenAI(api_key=api_key)
        elif provider == LLMProvider.ANTHROPIC:
            self.clients[provider] = anthropic.Anthropic(api_key=api_key)
        elif provider == LLMProvider.GOOGLE:
            genai.configure(api_key=api_key)
        elif provider == LLMProvider.GROQ:
            self.clients[provider] = groq.Groq(api_key=api_key)
    
    def chat(
        self,
        provider: LLMProvider,
        model: str,
        messages: List[Union[Message, Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request to the specified provider.
        
        Args:
            provider: The LLM provider to use
            model: The model name
            messages: List of messages in the conversation
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        
        Returns:
            LLMResponse object with standardized format
        """
        # Convert messages to standard format
        formatted_messages = self._format_messages(messages)
        
        if provider == LLMProvider.OPENAI:
            return self._chat_openai(model, formatted_messages, temperature, max_tokens, **kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            return self._chat_anthropic(model, formatted_messages, temperature, max_tokens, **kwargs)
        elif provider == LLMProvider.GOOGLE:
            return self._chat_google(model, formatted_messages, temperature, max_tokens, **kwargs)
        elif provider == LLMProvider.GROQ:
            return self._chat_groq(model, formatted_messages, temperature, max_tokens, **kwargs)
        elif provider == LLMProvider.OLLAMA:
            return self._chat_ollama(model, formatted_messages, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _format_messages(self, messages: List[Union[Message, Dict[str, str]]]) -> List[Dict[str, str]]:
        """Convert messages to standard dictionary format."""
        formatted = []
        for msg in messages:
            if isinstance(msg, Message):
                formatted.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg, dict):
                formatted.append(msg)
            else:
                raise ValueError(f"Invalid message format: {type(msg)}")
        return formatted
    
    def _chat_openai(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> LLMResponse:
        """Handle OpenAI chat completion."""
        client = self.clients[LLMProvider.OPENAI]
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        response = client.chat.completions.create(**params)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider=LLMProvider.OPENAI,
            model=model,
            usage=response.usage.model_dump() if response.usage else None,
            raw_response=response
        )
    
    def _chat_anthropic(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> LLMResponse:
        """Handle Anthropic chat completion."""
        client = self.clients[LLMProvider.ANTHROPIC]
        
        # Extract system message if present
        system_message = None
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        
        params = {
            "model": model,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1000,
            **kwargs
        }
        if system_message:
            params["system"] = system_message
        
        response = client.messages.create(**params)
        
        return LLMResponse(
            content=response.content[0].text,
            provider=LLMProvider.ANTHROPIC,
            model=model,
            usage=response.usage.__dict__ if hasattr(response, 'usage') else None,
            raw_response=response
        )
    
    def _chat_google(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> LLMResponse:
        """Handle Google Gemini chat completion."""
        # Initialize the model
        gemini_model = genai.GenerativeModel(model)
        
        # Convert messages to Gemini format
        chat_history = []
        user_message = ""
        
        for msg in messages:
            if msg["role"] == "system":
                # Gemini doesn't have explicit system messages, prepend to user message
                user_message = f"System: {msg['content']}\n\n" + user_message
            elif msg["role"] == "user":
                user_message = msg["content"]
            elif msg["role"] == "assistant":
                chat_history.append({
                    "role": "model",
                    "parts": [msg["content"]]
                })
        
        # Start chat with history
        chat = gemini_model.start_chat(history=chat_history)
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs
        )
        
        response = chat.send_message(user_message, generation_config=generation_config)
        
        return LLMResponse(
            content=response.text,
            provider=LLMProvider.GOOGLE,
            model=model,
            usage=None,  # Google doesn't provide detailed usage info in the same format
            raw_response=response
        )
    
    def _chat_groq(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> LLMResponse:
        """Handle Groq chat completion."""
        client = self.clients[LLMProvider.GROQ]
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        response = client.chat.completions.create(**params)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider=LLMProvider.GROQ,
            model=model,
            usage=response.usage.__dict__ if hasattr(response, 'usage') else None,
            raw_response=response
        )
    
    def _chat_ollama(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> LLMResponse:
        """Handle Ollama chat completion."""
        client = self.clients[LLMProvider.OLLAMA]
        
        # Ollama expects a different format
        params = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                **kwargs
            }
        }
        if max_tokens:
            params["options"]["num_predict"] = max_tokens
        
        response = client.chat(**params)
        
        return LLMResponse(
            content=response["message"]["content"],
            provider=LLMProvider.OLLAMA,
            model=model,
            usage=None,  # Ollama doesn't provide usage info in the same format
            raw_response=response
        )
    
    def simple_chat(
        self,
        provider: LLMProvider,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Simplified chat interface for single prompt/response.
        
        Args:
            provider: The LLM provider to use
            model: The model name
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            The response content as a string
        """
        messages = []
        if system_prompt:
            messages.append(Message("system", system_prompt))
        messages.append(Message("user", prompt))
        
        response = self.chat(provider, model, messages, temperature, max_tokens)
        return response.content
    
    def list_models(self, provider: LLMProvider) -> List[str]:
        """
        List available models for a provider.
        
        Args:
            provider: The LLM provider
        
        Returns:
            List of available model names
        """
        try:
            if provider == LLMProvider.OPENAI:
                models = self.clients[provider].models.list()
                return [model.id for model in models.data]
            elif provider == LLMProvider.OLLAMA:
                models = self.clients[provider].list()
                return [model["name"] for model in models["models"]]
            else:
                # For other providers, return common model names
                model_lists = {
                    LLMProvider.ANTHROPIC: [
                        "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307",
                        "claude-2.1",
                        "claude-2.0"
                    ],
                    LLMProvider.GOOGLE: [
                        "gemini-pro",
                        "gemini-pro-vision",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash"
                    ],
                    LLMProvider.GROQ: [
                        "llama2-70b-4096",
                        "mixtral-8x7b-32768",
                        "gemma-7b-it"
                    ]
                }
                return model_lists.get(provider, [])
        except Exception as e:
            print(f"Error listing models for {provider}: {e}")
            return []
    
    def is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is available and properly configured."""
        return provider in self.clients and self.clients[provider] is not None


# Example usage and convenience functions
def create_wrapper() -> LLMWrapper:
    """Create and return a new LLMWrapper instance."""
    return LLMWrapper()


def quick_chat(provider_name: str, model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Quick chat function for simple use cases.
    
    Args:
        provider_name: Name of the provider ("openai", "anthropic", "google", "groq", "ollama")
        model: Model name
        prompt: User prompt
        system_prompt: Optional system prompt
    
    Returns:
        Response content as string
    """
    wrapper = LLMWrapper()
    provider = LLMProvider(provider_name.lower())
    return wrapper.simple_chat(provider, model, prompt, system_prompt)