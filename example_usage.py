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
TokenUsage = ai_wrapper.TokenUsage

def main():
    # Create the wrapper instance
    wrapper = LLMWrapper()
    
    # Set API keys (replace with your actual keys or set as environment variables)
    # wrapper.set_api_key(LLMProvider.OPENAI, "your-openai-key")
    # wrapper.set_api_key(LLMProvider.ANTHROPIC, "your-anthropic-key")
    # wrapper.set_api_key(LLMProvider.GOOGLE, "your-google-key")
    # wrapper.set_api_key(LLMProvider.GROQ, "your-groq-key")
    
    # Example 1: Simple chat with token usage and cost tracking
    print("=== Example 1: Simple Chat with Cost Tracking ===")
    
    prompt = "Explain quantum computing in simple terms."
    responses = []
    
    # Try with different providers (uncomment the ones you have API keys for)
    
    # OpenAI
    try:
        if wrapper.is_provider_available(LLMProvider.OPENAI):
            response = wrapper.chat(
                LLMProvider.OPENAI,
                "gpt-3.5-turbo",
                [Message("system", "You are a helpful science teacher."),
                 Message("user", prompt)],
                temperature=0.7
            )
            responses.append(response)
            print(f"OpenAI Response: {response.content[:100]}...")
            if response.token_usage:
                usage = response.token_usage
                print(f"  Tokens - Input: {usage.input_tokens}, Output: {usage.output_tokens}")
                print(f"  Cost - Input: ${usage.input_cost:.6f}, Output: ${usage.output_cost:.6f}, Total: ${usage.total_cost:.6f}")
    except Exception as e:
        print(f"OpenAI error: {e}")
    
    # Anthropic
    try:
        if wrapper.is_provider_available(LLMProvider.ANTHROPIC):
            response = wrapper.chat(
                LLMProvider.ANTHROPIC,
                "claude-3-haiku-20240307",
                [Message("system", "You are a helpful science teacher."),
                 Message("user", prompt)],
                temperature=0.7,
                max_tokens=500
            )
            responses.append(response)
            print(f"Anthropic Response: {response.content[:100]}...")
            if response.token_usage:
                usage = response.token_usage
                print(f"  Tokens - Input: {usage.input_tokens}, Output: {usage.output_tokens}")
                print(f"  Cost - Input: ${usage.input_cost:.6f}, Output: ${usage.output_cost:.6f}, Total: ${usage.total_cost:.6f}")
    except Exception as e:
        print(f"Anthropic error: {e}")
    
    # Ollama (local - free)
    try:
        if wrapper.is_provider_available(LLMProvider.OLLAMA):
            response = wrapper.chat(
                LLMProvider.OLLAMA,
                "llama2",  # Make sure you have llama2 installed in Ollama
                [Message("system", "You are a helpful science teacher."),
                 Message("user", prompt)],
                temperature=0.7
            )
            responses.append(response)
            print(f"Ollama Response: {response.content[:100]}...")
            if response.token_usage:
                usage = response.token_usage
                print(f"  Tokens - Input: {usage.input_tokens}, Output: {usage.output_tokens}")
                print(f"  Cost - Input: ${usage.input_cost:.6f}, Output: ${usage.output_cost:.6f}, Total: ${usage.total_cost:.6f} (Local model - Free!)")
    except Exception as e:
        print(f"Ollama error: {e}")
    
    # Generate cost summary
    if responses:
        print("\n=== Cost Summary ===")
        summary = wrapper.get_cost_summary(responses)
        print(f"Total Cost: ${summary['total_cost']:.6f}")
        print(f"Total Tokens: {summary['total_tokens']} (Input: {summary['total_input_tokens']}, Output: {summary['total_output_tokens']})")
        
        print("\nBy Provider:")
        for provider, data in summary['by_provider'].items():
            print(f"  {provider}: ${data['cost']:.6f} ({data['requests']} requests, {data['total_tokens']} tokens)")
        
        print("\nBy Model:")
        for model, data in summary['by_model'].items():
            print(f"  {model}: ${data['cost']:.6f} ({data['requests']} requests, {data['total_tokens']} tokens)")
    
    print("\n=== Example 2: Multi-turn Conversation with Cost Tracking ===")
    
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
            if response.token_usage:
                usage = response.token_usage
                print(f"Token Usage: {usage.input_tokens} input + {usage.output_tokens} output = {usage.total_tokens} total")
                print(f"Cost Breakdown: ${usage.input_cost:.6f} + ${usage.output_cost:.6f} = ${usage.total_cost:.6f}")
                print(f"Rate: ${usage.cost_per_input_token*1000000:.2f}/M input tokens, ${usage.cost_per_output_token*1000000:.2f}/M output tokens")
    except Exception as e:
        print(f"Multi-turn conversation error: {e}")
    
    print("\n=== Example 3: Cost Comparison Between Providers ===")
    cost_comparison_responses = []
    test_prompt = "Write a short poem about programming."
    
    providers_to_test = [
        (LLMProvider.OPENAI, "gpt-3.5-turbo"),
        (LLMProvider.ANTHROPIC, "claude-3-haiku-20240307"),
        (LLMProvider.OLLAMA, "llama2"),
    ]
    
    for provider, model in providers_to_test:
        try:
            if wrapper.is_provider_available(provider):
                response = wrapper.chat(provider, model, [Message("user", test_prompt)])
                cost_comparison_responses.append(response)
                print(f"\n{provider.value} ({model}):")
                print(f"Response: {response.content[:100]}...")
                if response.token_usage:
                    usage = response.token_usage
                    print(f"Cost: ${usage.total_cost:.6f} ({usage.total_tokens} tokens)")
        except Exception as e:
            print(f"{provider.value} error: {e}")
    
    if cost_comparison_responses:
        print("\n=== Cost Comparison Summary ===")
        comparison_summary = wrapper.get_cost_summary(cost_comparison_responses)
        print(f"Total across all providers: ${comparison_summary['total_cost']:.6f}")
        
        # Sort by cost
        sorted_providers = sorted(comparison_summary['by_provider'].items(), key=lambda x: x[1]['cost'])
        print("\nRanked by cost (cheapest first):")
        for provider, data in sorted_providers:
            cost_per_token = data['cost'] / data['total_tokens'] if data['total_tokens'] > 0 else 0
            print(f"  {provider}: ${data['cost']:.6f} (${cost_per_token*1000:.3f} per 1K tokens)")
    
    print("\n=== Example 4: List Available Models ===")
    
    # Example 4: List available models
    for provider in LLMProvider:
        try:
            if wrapper.is_provider_available(provider):
                models = wrapper.list_models(provider)
                print(f"{provider.value} models: {models[:5]}...")  # Show first 5 models
        except Exception as e:
            print(f"Error listing {provider.value} models: {e}")


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
