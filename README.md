# AI Wrapper Library

A unified Python library that provides a consistent interface for multiple Large Language Model (LLM) providers.

## Supported Providers

- **OpenAI** (GPT models)
- **Anthropic** (Claude models)
- **Google** (Gemini models)
- **Groq** (Fast inference)
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
class LLMResponse:
    content: str                    # The generated text
    provider: LLMProvider          # Which provider was used
    model: str                     # Which model was used
    usage: Optional[Dict]          # Token usage information
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

#### `is_provider_available(provider: LLMProvider)`
Check if a provider is properly configured and available.

### Convenience Functions

#### `quick_chat(provider_name, model, prompt, system_prompt=None)`
Quick one-liner for simple chat interactions.

#### `create_wrapper()`
Factory function to create a new LLMWrapper instance.

## Usage Examples

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

## Configuration

### Environment Variables
You can set API keys using environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"
```

### Manual Configuration
```python
wrapper = LLMWrapper()
wrapper.set_api_key(LLMProvider.OPENAI, "your-openai-key")
wrapper.set_api_key(LLMProvider.ANTHROPIC, "your-anthropic-key")
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

### Ollama
- No API key required
- Runs locally
- Requires Ollama to be installed and running
- Different parameter names (e.g., `num_predict` instead of `max_tokens`)

## License

This library is provided as-is for educational and development purposes.
