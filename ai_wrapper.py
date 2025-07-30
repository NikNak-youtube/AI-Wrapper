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
    XAI = "xai"  # xAI Grok
    OLLAMA = "ollama"
    LLAMA = "llama"  # Llama API


@dataclass
class TokenUsage:
    """Detailed token usage and cost information."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float  # Cost in USD
    output_cost: float  # Cost in USD
    total_cost: float  # Cost in USD
    cost_per_input_token: float  # Cost per token in USD
    cost_per_output_token: float  # Cost per token in USD


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    content: str
    provider: LLMProvider
    model: str
    token_usage: Optional[TokenUsage] = None
    usage: Optional[Dict[str, Any]] = None  # Raw usage data from provider
    raw_response: Optional[Any] = None


@dataclass
class Message:
    """Standardized message format."""
    role: str  # "user", "assistant", "system"
    content: str


class LLMWrapper:
    """
    Unified wrapper for multiple LLM providers.

    Supports: OpenAI, Anthropic, Google Gemini, Groq, xAI Grok, and Ollama
    """

    # Pricing information (per 1M tokens in USD) - Updated as of July 2025
    PRICING = {
        LLMProvider.OPENAI: {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
            "gpt-4-0125-preview": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        },
        LLMProvider.ANTHROPIC: {
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-2.1": {"input": 8.0, "output": 24.0},
            "claude-2.0": {"input": 8.0, "output": 24.0},
            "claude-instant-1.2": {"input": 0.8, "output": 2.4},
        },
        LLMProvider.GOOGLE: {
            "gemini-pro": {"input": 0.5, "output": 1.5},
            "gemini-pro-vision": {"input": 0.5, "output": 1.5},
            "gemini-1.5-pro": {"input": 3.5, "output": 10.5},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.3},
        },
        LLMProvider.GROQ: {
            # Groq often has free tiers or very low costs
            "llama2-70b-4096": {"input": 0.0, "output": 0.0},
            "mixtral-8x7b-32768": {"input": 0.0, "output": 0.0},
            "gemma-7b-it": {"input": 0.0, "output": 0.0},
            "llama3-8b-8192": {"input": 0.05, "output": 0.08},
            "llama3-70b-8192": {"input": 0.59, "output": 0.79},
        },
        LLMProvider.XAI: {
            # xAI Grok pricing (as of July 2025)
            "grok-beta": {"input": 5.0, "output": 15.0},
            "grok-vision-beta": {"input": 5.0, "output": 15.0},
        },
        LLMProvider.OLLAMA: {
            # Ollama is local, so no cost
            "default": {"input": 0.0, "output": 0.0},
        },
        LLMProvider.LLAMA: {
            # Llama API pricing (placeholder - update with actual pricing)
            "Llama-4-Scout-17B-16E-Instruct-FP8": {"input": 0.5, "output": 1.5},
            "default": {"input": 0.5, "output": 1.5}
        }
    }

    def __init__(self):
        self.clients = {}
        self._initialize_clients()

    def _calculate_cost(self, provider: LLMProvider, model: str, input_tokens: int, output_tokens: int) -> TokenUsage:
        """Calculate token usage and costs."""
        # Get pricing for the model
        provider_pricing = self.PRICING.get(provider, {})

        # Handle model variations and fallbacks
        model_pricing = None
        if model in provider_pricing:
            model_pricing = provider_pricing[model]
        else:
            # Try to find a matching model by checking if the model name contains known models
            for known_model, pricing in provider_pricing.items():
                if known_model in model or model in known_model:
                    model_pricing = pricing
                    break

            # If still no match, use default pricing or zero for local models
            if not model_pricing:
                if provider == LLMProvider.OLLAMA:
                    model_pricing = {"input": 0.0, "output": 0.0}
                else:
                    # Use a conservative default pricing
                    model_pricing = {"input": 1.0, "output": 2.0}

        # Calculate costs (pricing is per 1M tokens)
        cost_per_input_token = model_pricing["input"] / 1_000_000
        cost_per_output_token = model_pricing["output"] / 1_000_000

        input_cost = input_tokens * cost_per_input_token
        output_cost = output_tokens * cost_per_output_token
        total_cost = input_cost + output_cost
        total_tokens = input_tokens + output_tokens

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            cost_per_input_token=cost_per_input_token,
            cost_per_output_token=cost_per_output_token
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Very rough estimation: ~4 characters per token for most models
        # This is not precise but gives a reasonable estimate when exact counts aren't available
        return max(1, len(text) // 4)

    def _clean_llama_response(self, response: str) -> str:
        """Clean Llama API response by removing internal tokens and their content."""
        if not response:
            return response

        import re

        # Remove content between paired tokens (including the tokens themselves)
        patterns_to_remove = [
            r'<\|tool_calls_start_id\|>.*?<\|end_tool_call_start_id\|>',
            r'<\|tool_call_start\|>.*?<\|tool_call_end\|>',
            r'<\|start_header_id\|>.*?<\|end_header_id\|>',
            r'<\|im_start\|>.*?<\|im_end\|>',
        ]

        # Also remove standalone tokens that might not have pairs
        standalone_tokens = [
            '<|tool_calls_start_id|>',
            '<|end_tool_call_start_id|>',
            '<|end_tool_call_id|>',
            '<|tool_call_start|>',
            '<|tool_call_end|>',
            '<|im_start|>',
            '<|im_end|>'
        ]

        cleaned = response

        # Remove paired content first
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)

        # Remove any remaining standalone tokens
        for token in standalone_tokens:
            cleaned = cleaned.replace(token, '')

        # Clean up extra whitespace and newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()

        return cleaned

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
            # xAI client uses OpenAI-compatible API
            self.clients[LLMProvider.XAI] = openai.OpenAI(
                api_key="",  # Will be set when user calls set_api_key
                base_url="https://api.x.ai/v1"
            )
        except Exception as e:
            print(f"Warning: xAI client initialization failed: {e}")

        try:
            # Ollama client is initialized when needed
            self.clients[LLMProvider.OLLAMA] = ollama
        except Exception as e:
            print(f"Warning: Ollama client initialization failed: {e}")

        try:
            # Llama client uses OpenAI-compatible API
            self.clients[LLMProvider.LLAMA] = openai.OpenAI(
                api_key="",  # Will be set when user calls set_api_key
                base_url="https://api.llama.com/compat/v1"
            )
        except Exception as e:
            print(f"Warning: Llama client initialization failed: {e}")

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
        elif provider == LLMProvider.XAI:
            self.clients[provider] = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        elif provider == LLMProvider.LLAMA:
            self.clients[provider] = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.llama.com/compat/v1"
            )

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
        elif provider == LLMProvider.XAI:
            return self._chat_xai(model, formatted_messages, temperature, max_tokens, **kwargs)
        elif provider == LLMProvider.OLLAMA:
            return self._chat_ollama(model, formatted_messages, temperature, max_tokens, **kwargs)
        elif provider == LLMProvider.LLAMA:
            return self._chat_llama(model, formatted_messages, temperature, max_tokens, **kwargs)
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

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Calculate costs
        token_usage = self._calculate_cost(LLMProvider.OPENAI, model, input_tokens, output_tokens)

        return LLMResponse(
            content=response.choices[0].message.content,
            provider=LLMProvider.OPENAI,
            model=model,
            token_usage=token_usage,
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

        # Extract token usage
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        # Calculate costs
        token_usage = self._calculate_cost(LLMProvider.ANTHROPIC, model, input_tokens, output_tokens)

        return LLMResponse(
            content=response.content[0].text,
            provider=LLMProvider.ANTHROPIC,
            model=model,
            token_usage=token_usage,
            usage=response.usage.__dict__ if hasattr(response, 'usage') else None,
            raw_response=response
        )

    def _chat_google(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> LLMResponse:
        """Handle Google Gemini chat completion."""
        # Initialize the model
        gemini_model = genai.GenerativeModel(model)

        # Convert messages to Gemini format and estimate input tokens
        chat_history = []
        user_message = ""
        input_text = ""

        for msg in messages:
            if msg["role"] == "system":
                # Gemini doesn't have explicit system messages, prepend to user message
                system_text = f"System: {msg['content']}\n\n"
                user_message = system_text + user_message
                input_text += system_text
            elif msg["role"] == "user":
                user_message = msg["content"]
                input_text += msg["content"]
            elif msg["role"] == "assistant":
                chat_history.append({
                    "role": "model",
                    "parts": [msg["content"]]
                })
                input_text += msg["content"]

        # Start chat with history
        chat = gemini_model.start_chat(history=chat_history)

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs
        )

        response = chat.send_message(user_message, generation_config=generation_config)

        # Estimate token usage (Google doesn't always provide exact counts)
        input_tokens = self._estimate_tokens(input_text + user_message)
        output_tokens = self._estimate_tokens(response.text)

        # Try to get actual usage if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            if hasattr(response.usage_metadata, 'prompt_token_count'):
                input_tokens = response.usage_metadata.prompt_token_count
            if hasattr(response.usage_metadata, 'candidates_token_count'):
                output_tokens = response.usage_metadata.candidates_token_count

        # Calculate costs
        token_usage = self._calculate_cost(LLMProvider.GOOGLE, model, input_tokens, output_tokens)

        return LLMResponse(
            content=response.text,
            provider=LLMProvider.GOOGLE,
            model=model,
            token_usage=token_usage,
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

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # If no usage data, estimate
        if not usage:
            input_text = " ".join([msg["content"] for msg in messages])
            input_tokens = self._estimate_tokens(input_text)
            output_tokens = self._estimate_tokens(response.choices[0].message.content)

        # Calculate costs
        token_usage = self._calculate_cost(LLMProvider.GROQ, model, input_tokens, output_tokens)

        return LLMResponse(
            content=response.choices[0].message.content,
            provider=LLMProvider.GROQ,
            model=model,
            token_usage=token_usage,
            usage=response.usage.__dict__ if hasattr(response, 'usage') else None,
            raw_response=response
        )

    def _chat_xai(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> LLMResponse:
        """Handle xAI Grok chat completion."""
        client = self.clients[LLMProvider.XAI]

        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            params["max_tokens"] = max_tokens

        response = client.chat.completions.create(**params)

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # If no usage data, estimate
        if not usage:
            input_text = " ".join([msg["content"] for msg in messages])
            input_tokens = self._estimate_tokens(input_text)
            output_tokens = self._estimate_tokens(response.choices[0].message.content)

        # Calculate costs
        token_usage = self._calculate_cost(LLMProvider.XAI, model, input_tokens, output_tokens)

        return LLMResponse(
            content=response.choices[0].message.content,
            provider=LLMProvider.XAI,
            model=model,
            token_usage=token_usage,
            usage=response.usage.model_dump() if response.usage else None,
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

        # Estimate token usage (Ollama doesn't provide token counts)
        input_text = " ".join([msg["content"] for msg in messages])
        input_tokens = self._estimate_tokens(input_text)
        output_tokens = self._estimate_tokens(response["message"]["content"])

        # Check if Ollama provided any token information
        if "prompt_eval_count" in response:
            input_tokens = response["prompt_eval_count"]
        if "eval_count" in response:
            output_tokens = response["eval_count"]

        # Calculate costs (Ollama is local, so cost is 0)
        token_usage = self._calculate_cost(LLMProvider.OLLAMA, model, input_tokens, output_tokens)

        return LLMResponse(
            content=response["message"]["content"],
            provider=LLMProvider.OLLAMA,
            model=model,
            token_usage=token_usage,
            usage=None,  # Ollama doesn't provide usage info in the same format
            raw_response=response
        )

    def _chat_llama(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> LLMResponse:
        """Handle Llama chat completion."""
        client = self.clients[LLMProvider.LLAMA]

        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            params["max_tokens"] = max_tokens

        response = client.chat.completions.create(**params)

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # If no usage data, estimate
        if not usage:
            input_text = " ".join([msg["content"] for msg in messages])
            input_tokens = self._estimate_tokens(input_text)
            output_tokens = self._estimate_tokens(response.choices[0].message.content)

        # Calculate costs
        token_usage = self._calculate_cost(LLMProvider.LLAMA, model, input_tokens, output_tokens)

        # Clean response content for Llama API
        content = response.choices[0].message.content
        content = self._clean_llama_response(content)

        return LLMResponse(
            content=content,
            provider=LLMProvider.LLAMA,
            model=model,
            token_usage=token_usage,
            usage=response.usage.model_dump() if response.usage else None,
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
                    ],
                    LLMProvider.XAI: [
                        "grok-beta",
                        "grok-vision-beta"
                    ],
                    LLMProvider.LLAMA: [
                        "Llama-4-Scout-17B-16E-Instruct-FP8"
                    ]
                }
                return model_lists.get(provider, [])
        except Exception as e:
            print(f"Error listing models for {provider}: {e}")
            return []

    def is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is available and properly configured."""
        return provider in self.clients and self.clients[provider] is not None

    def get_embeddings(self, provider: LLMProvider, text: str, model: Optional[str] = None) -> List[float]:
        """
        Get embeddings for text using the specified provider.

        Args:
            provider: The LLM provider to use
            text: Text to get embeddings for
            model: Optional model override (uses default if not specified)

        Returns:
            List of embedding values (floats)

        Raises:
            ValueError: If provider is not configured or doesn't support embeddings
            Exception: If embedding generation fails
        """
        # Get the client
        client = self.clients.get(provider)
        if client is None:
            raise ValueError(f"Provider {provider.value} is not configured")

        # OpenAI-compatible providers (OpenAI, Groq, xAI, Llama)
        if provider in [LLMProvider.OPENAI, LLMProvider.GROQ, LLMProvider.XAI, LLMProvider.LLAMA]:
            if model is None:
                model = "text-embedding-3-small"
            response = client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding

        elif provider == LLMProvider.OLLAMA:
            # Ollama has embedding models like nomic-embed-text
            if model is None:
                model = "nomic-embed-text"  # Default Ollama embedding model
            response = client.embeddings(
                model=model,
                prompt=text
            )
            return response["embedding"]

        else:
            # Anthropic and Google don't have native embedding APIs
            raise ValueError(f"Provider {provider.value} does not support embeddings")

    def get_cost_summary(self, responses: List[LLMResponse]) -> Dict[str, Any]:
        """
        Generate a cost summary from multiple responses.

        Args:
            responses: List of LLMResponse objects

        Returns:
            Dictionary with cost breakdown by provider and totals
        """
        summary = {
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "by_provider": {},
            "by_model": {}
        }

        for response in responses:
            if response.token_usage:
                usage = response.token_usage
                provider_name = response.provider.value
                model_name = response.model

                # Update totals
                summary["total_cost"] += usage.total_cost
                summary["total_input_tokens"] += usage.input_tokens
                summary["total_output_tokens"] += usage.output_tokens
                summary["total_tokens"] += usage.total_tokens

                # Update by provider
                if provider_name not in summary["by_provider"]:
                    summary["by_provider"][provider_name] = {
                        "cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "requests": 0
                    }

                summary["by_provider"][provider_name]["cost"] += usage.total_cost
                summary["by_provider"][provider_name]["input_tokens"] += usage.input_tokens
                summary["by_provider"][provider_name]["output_tokens"] += usage.output_tokens
                summary["by_provider"][provider_name]["total_tokens"] += usage.total_tokens
                summary["by_provider"][provider_name]["requests"] += 1

                # Update by model
                if model_name not in summary["by_model"]:
                    summary["by_model"][model_name] = {
                        "cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "requests": 0
                    }

                summary["by_model"][model_name]["cost"] += usage.total_cost
                summary["by_model"][model_name]["input_tokens"] += usage.input_tokens
                summary["by_model"][model_name]["output_tokens"] += usage.output_tokens
                summary["by_model"][model_name]["total_tokens"] += usage.total_tokens
                summary["by_model"][model_name]["requests"] += 1

        return summary


# Example usage and convenience functions
def create_wrapper() -> LLMWrapper:
    """Create and return a new LLMWrapper instance."""
    return LLMWrapper()


def quick_chat(provider_name: str, model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Quick chat function for simple use cases.

    Args:
        provider_name: Name of the provider ("openai", "anthropic", "google", "groq", "xai", "ollama", "llama")
        model: Model name
        prompt: User prompt
        system_prompt: Optional system prompt

    Returns:
        Response content as string
    """
    wrapper = LLMWrapper()
    provider = LLMProvider(provider_name.lower())
    return wrapper.simple_chat(provider, model, prompt, system_prompt)
