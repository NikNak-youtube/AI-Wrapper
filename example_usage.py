"""
Example usage of the AI Wrapper library.

This file demonstrates how to use the unified LLM wrapper to interact
with different AI providers using the same interface.
"""

# Import from the AI Wrapper module (note: the filename has a space)
import sys
import importlib.util
import os

# Load the module with a space in its name
spec = importlib.util.spec_from_file_location("ai_wrapper", "AI Wrapper.py")
ai_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_wrapper)

# Import the classes and functions
LLMWrapper = ai_wrapper.LLMWrapper
LLMProvider = ai_wrapper.LLMProvider
Message = ai_wrapper.Message
quick_chat = ai_wrapper.quick_chat

def main():
    # Create the wrapper instance
    wrapper = LLMWrapper()
    
    # Set API keys (replace with your actual keys or set as environment variables)
    # wrapper.set_api_key(LLMProvider.OPENAI, "your-openai-key")
    # wrapper.set_api_key(LLMProvider.ANTHROPIC, "your-anthropic-key")
    # wrapper.set_api_key(LLMProvider.GOOGLE, "your-google-key")
    # wrapper.set_api_key(LLMProvider.GROQ, "your-groq-key")
    
    # Example 1: Simple chat with different providers
    print("=== Example 1: Simple Chat ===")
    
    prompt = "Explain quantum computing in simple terms."
    
    # Try with different providers (uncomment the ones you have API keys for)
    
    # OpenAI
    try:
        if wrapper.is_provider_available(LLMProvider.OPENAI):
            response = wrapper.simple_chat(
                LLMProvider.OPENAI,
                "gpt-3.5-turbo",
                prompt,
                system_prompt="You are a helpful science teacher."
            )
            print(f"OpenAI Response: {response[:200]}...")
    except Exception as e:
        print(f"OpenAI error: {e}")
    
    # Anthropic
    try:
        if wrapper.is_provider_available(LLMProvider.ANTHROPIC):
            response = wrapper.simple_chat(
                LLMProvider.ANTHROPIC,
                "claude-3-haiku-20240307",
                prompt,
                system_prompt="You are a helpful science teacher."
            )
            print(f"Anthropic Response: {response[:200]}...")
    except Exception as e:
        print(f"Anthropic error: {e}")
    
    # Ollama (local)
    try:
        if wrapper.is_provider_available(LLMProvider.OLLAMA):
            response = wrapper.simple_chat(
                LLMProvider.OLLAMA,
                "llama2",  # Make sure you have llama2 installed in Ollama
                prompt,
                system_prompt="You are a helpful science teacher."
            )
            print(f"Ollama Response: {response[:200]}...")
    except Exception as e:
        print(f"Ollama error: {e}")
    
    print("\n=== Example 2: Multi-turn Conversation ===")
    
    # Example 2: Multi-turn conversation
    messages = [
        Message("system", "You are a helpful coding assistant."),
        Message("user", "How do I create a Python class?"),
        Message("assistant", "To create a Python class, use the 'class' keyword followed by the class name..."),
        Message("user", "Can you show me an example with inheritance?")
    ]
    
    try:
        if wrapper.is_provider_available(LLMProvider.OPENAI):
            response = wrapper.chat(
                LLMProvider.OPENAI,
                "gpt-3.5-turbo",
                messages,
                temperature=0.7,
                max_tokens=500
            )
            print(f"Multi-turn response: {response.content[:200]}...")
            print(f"Usage: {response.usage}")
    except Exception as e:
        print(f"Multi-turn conversation error: {e}")
    
    print("\n=== Example 3: List Available Models ===")
    
    # Example 3: List available models
    for provider in LLMProvider:
        try:
            if wrapper.is_provider_available(provider):
                models = wrapper.list_models(provider)
                print(f"{provider.value} models: {models[:5]}...")  # Show first 5 models
        except Exception as e:
            print(f"Error listing {provider.value} models: {e}")
    
    print("\n=== Example 4: Quick Chat Function ===")
    
    # Example 4: Using the quick_chat convenience function
    try:
        response = quick_chat(
            "openai",
            "gpt-3.5-turbo",
            "What is the capital of France?",
            system_prompt="Answer concisely."
        )
        print(f"Quick chat response: {response}")
    except Exception as e:
        print(f"Quick chat error: {e}")


def compare_providers():
    """
    Compare responses from different providers for the same prompt.
    """
    print("\n=== Provider Comparison ===")
    
    wrapper = LLMWrapper()
    prompt = "Write a haiku about programming."
    
    providers_to_test = [
        (LLMProvider.OPENAI, "gpt-3.5-turbo"),
        (LLMProvider.ANTHROPIC, "claude-3-haiku-20240307"),
        (LLMProvider.OLLAMA, "llama2"),
    ]
    
    for provider, model in providers_to_test:
        try:
            if wrapper.is_provider_available(provider):
                response = wrapper.simple_chat(provider, model, prompt)
                print(f"\n{provider.value} ({model}):")
                print(response)
        except Exception as e:
            print(f"{provider.value} error: {e}")


if __name__ == "__main__":
    main()
    compare_providers()
