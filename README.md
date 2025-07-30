# AI Wrapper Library

A unified Python library that provides a consistent interface for multiple Large Language Model (LLM) providers.

## Supported Providers

- **OpenAI** (GPT models)
- **Anthropic** (Claude models)
- **Google** (Gemini models)
- **Groq** (Fast inference)
- **xAI** (Grok models)
- **Ollama** (Local models)

## Installation

Install the required dependencies:

```bash
pip install openai anthropic google-generativeai groq ollama-python
```

## Quick Start

```python
from AI_Wrapper import LLMWrapper, LLMProvider, quick_chat

# Simple usage
response = quick_chat("openai", "gpt-3.5-turbo", "Hello, world!")
print(response)

# Advanced usage
wrapper = LLMWrapper()
wrapper.set_api_key(LLMProvider.OPENAI, "your-api-key")

response = wrapper.simple_chat(
    LLMProvider.OPENAI,
    "gpt-3.5-turbo",
    "Explain quantum computing",
    system_prompt="You are a helpful science teacher"
)
```

## Features

### Unified Interface
All providers use the same method signatures and return standardized response objects.

### Message Format
```python
from AI_Wrapper import Message

messages = [
    Message("system", "You are a helpful assistant"),
    Message("user", "What is Python?"),
    Message("assistant", "Python is a programming language..."),
    Message("user", "Tell me more")
]
```

### Response Format
```python
@dataclass
class TokenUsage:
    input_tokens: int              # Number of input tokens
    output_tokens: int             # Number of output tokens
    total_tokens: int              # Total tokens used
    input_cost: float              # Cost for input tokens (USD)
    output_cost: float             # Cost for output tokens (USD)
    total_cost: float              # Total cost (USD)
    cost_per_input_token: float    # Rate per input token (USD)
    cost_per_output_token: float   # Rate per output token (USD)

@dataclass
class LLMResponse:
    content: str                    # The generated text
    provider: LLMProvider          # Which provider was used
    model: str                     # Which model was used
    token_usage: TokenUsage        # Detailed token and cost info
    usage: Optional[Dict]          # Raw usage data from provider
    raw_response: Optional[Any]    # Original provider response
```

## API Reference

### LLMWrapper Class

#### `__init__()`
Creates a new wrapper instance and initializes all available clients.

#### `set_api_key(provider: LLMProvider, api_key: str)`
Set the API key for a specific provider.

#### `chat(provider, model, messages, temperature=0.7, max_tokens=None, **kwargs)`
Send a chat completion request with full control over parameters.

#### `simple_chat(provider, model, prompt, system_prompt=None, temperature=0.7, max_tokens=None)`
Simplified interface for single prompt/response interactions.

#### `list_models(provider: LLMProvider)`
List available models for a provider.

#### `get_embeddings(provider: LLMProvider, text: str, model: Optional[str] = None)`
Get embeddings for text using the specified provider.

- **Supported providers**: OpenAI, Groq, xAI, Ollama
- **Default models**: 
  - OpenAI/Groq/xAI: `text-embedding-3-small`
  - Ollama: `nomic-embed-text`
- **Returns**: List of floats representing the embedding vector
- **Raises**: `ValueError` if provider doesn't support embeddings

#### `get_cost_summary(responses: List[LLMResponse])`

Generate a comprehensive cost summary from multiple responses.

## Token Usage and Cost Tracking

The library automatically tracks token usage and calculates costs for each request:

```python
response = wrapper.chat(LLMProvider.OPENAI, "gpt-3.5-turbo", messages)

if response.token_usage:
    usage = response.token_usage
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Input cost: ${usage.input_cost:.6f}")
    print(f"Output cost: ${usage.output_cost:.6f}")
    print(f"Total cost: ${usage.total_cost:.6f}")
```

### Cost Summary Across Multiple Requests

```python
responses = []
responses.append(wrapper.chat(LLMProvider.OPENAI, "gpt-3.5-turbo", messages1))
responses.append(wrapper.chat(LLMProvider.ANTHROPIC, "claude-3-haiku-20240307", messages2))

summary = wrapper.get_cost_summary(responses)
print(f"Total cost: ${summary['total_cost']:.6f}")
print(f"Total tokens: {summary['total_tokens']}")

# Cost breakdown by provider
for provider, data in summary['by_provider'].items():
    print(f"{provider}: ${data['cost']:.6f} ({data['total_tokens']} tokens)")
```

### Pricing Information

The library includes up-to-date pricing for all supported providers:

- **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4-turbo, etc.
- **Anthropic**: Claude-3 (Opus, Sonnet, Haiku), Claude-2
- **Google**: Gemini Pro, Gemini 1.5 Pro/Flash
- **Groq**: Often free or very low cost
- **xAI**: Grok models with competitive pricing
- **Ollama**: Free (local execution)

Pricing is automatically updated based on the model used and calculates costs per token.

## Usage Examples

### Basic Chat with Cost Tracking

```python
wrapper = LLMWrapper()
response = wrapper.chat(
    LLMProvider.OPENAI,
    "gpt-3.5-turbo",
    [Message("user", "Hello, world!")]
)

print(f"Response: {response.content}")
if response.token_usage:
    print(f"Cost: ${response.token_usage.total_cost:.6f}")
```

### Basic Chat
```python
wrapper = LLMWrapper()
response = wrapper.simple_chat(
    LLMProvider.OPENAI,
    "gpt-3.5-turbo",
    "Write a haiku about programming"
)
print(response)
```

### Multi-turn Conversation
```python
messages = [
    Message("system", "You are a coding tutor"),
    Message("user", "How do I create a Python function?"),
    Message("assistant", "You can create a function using the 'def' keyword..."),
    Message("user", "Can you show me an example?")
]

response = wrapper.chat(
    LLMProvider.ANTHROPIC,
    "claude-3-haiku-20240307",
    messages,
    temperature=0.7
)
print(response.content)
```

### Compare Multiple Providers
```python
prompt = "Explain machine learning in one sentence"

providers = [
    (LLMProvider.OPENAI, "gpt-3.5-turbo"),
    (LLMProvider.ANTHROPIC, "claude-3-haiku-20240307"),
    (LLMProvider.OLLAMA, "llama2")
]

for provider, model in providers:
    if wrapper.is_provider_available(provider):
        response = wrapper.simple_chat(provider, model, prompt)
        print(f"{provider.value}: {response}")
```

### Working with Local Models (Ollama)
```python
# Make sure Ollama is running and you have models installed
# ollama pull llama2

response = wrapper.simple_chat(
    LLMProvider.OLLAMA,
    "llama2",
    "What is the capital of France?"
)
```

### Using Embeddings
```python
# Get embeddings from OpenAI
embeddings = wrapper.get_embeddings(
    LLMProvider.OPENAI,
    "The quick brown fox jumps over the lazy dog"
)
print(f"Embedding dimension: {len(embeddings)}")

# Use a specific embedding model
embeddings = wrapper.get_embeddings(
    LLMProvider.OPENAI,
    "Sample text",
    model="text-embedding-ada-002"
)

# Try Ollama embeddings (requires embedding model like nomic-embed-text)
# ollama pull nomic-embed-text
try:
    embeddings = wrapper.get_embeddings(
        LLMProvider.OLLAMA,
        "Local embeddings test"
    )
except ValueError as e:
    print(f"Ollama embeddings not available: {e}")

# Note: Anthropic and Google don't support embeddings
# This will raise ValueError
try:
    embeddings = wrapper.get_embeddings(LLMProvider.ANTHROPIC, "test")
except ValueError as e:
    print(e)  # "Provider anthropic does not support embeddings"
```

## Configuration

### Environment Variables
You can set API keys using environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"
export XAI_API_KEY="your-xai-key"
```

### Manual Configuration
```python
wrapper = LLMWrapper()
wrapper.set_api_key(LLMProvider.OPENAI, "your-openai-key")
wrapper.set_api_key(LLMProvider.ANTHROPIC, "your-anthropic-key")
wrapper.set_api_key(LLMProvider.XAI, "your-xai-key")
```

## Common Model Names

### OpenAI
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Anthropic
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### Google
- `gemini-pro`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

### Groq
- `llama2-70b-4096`
- `mixtral-8x7b-32768`
- `gemma-7b-it`

### xAI
- `grok-beta`
- `grok-vision-beta`

### Ollama
- `llama2`
- `codellama`
- `mistral`
- (Any model you have installed locally)

## Error Handling

The library includes built-in error handling:

```python
try:
    response = wrapper.simple_chat(
        LLMProvider.OPENAI,
        "gpt-3.5-turbo",
        "Hello!"
    )
    print(response)
except Exception as e:
    print(f"Error: {e}")
```

## Provider-Specific Notes

### OpenAI
- Requires API key
- Supports all standard parameters
- Provides detailed usage information

### Anthropic
- Requires API key
- System messages are handled specially
- Has a required `max_tokens` parameter

### Google (Gemini)
- Requires API key
- System messages are prepended to user messages
- Different message format internally

### Groq
- Requires API key
- Fast inference
- Limited model selection

### xAI
- Requires API key
- Uses OpenAI-compatible API
- Grok models with real-time information access

### Ollama
- No API key required
- Runs locally
- Requires Ollama to be installed and running
- Different parameter names (e.g., `num_predict` instead of `max_tokens`)

## License

This library is provided as-is for educational and development purposes.
