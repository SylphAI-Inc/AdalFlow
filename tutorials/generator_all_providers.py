# This doc shows how to use all different providers in the Generator class.

import adalflow as adal


def use_all_providers():
    openai_llm = adal.Generator(
        model_client=adal.OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo"},
    )
    groq_llm = adal.Generator(
        model_client=adal.GroqAPIClient(),
        model_kwargs={"model": "llama3-8b-8192"},
    )
    anthropic_llm = adal.Generator(
        model_client=adal.AnthropicAPIClient(),
        model_kwargs={"model": "claude-3-5-sonnet-20241022"},
    )
    google_gen_ai_llm = adal.Generator(
        model_client=adal.GoogleGenAIClient(),
        model_kwargs={"model": "gemini-1.0-pro"},
    )
    ollama_llm = adal.Generator(
        model_client=adal.OllamaClient(),
        model_kwargs={"model": "llama3.2:1b"},
    )
    # need to run ollama pull llama3.2:1b first to use this model

    # aws_bedrock_llm = adal.Generator(
    #     model_client=adal.BedrockAPIClient(),
    #     model_kwargs={"modelId": "amazon.mistral.instruct-7b"},
    # )

    prompt_kwargs = {"input_str": "What is the meaning of life in one sentence?"}

    openai_response = openai_llm(prompt_kwargs)
    groq_response = groq_llm(prompt_kwargs)
    anthropic_response = anthropic_llm(prompt_kwargs)
    google_gen_ai_response = google_gen_ai_llm(prompt_kwargs)
    ollama_response = ollama_llm(prompt_kwargs)
    # aws_bedrock_llm_response = aws_bedrock_llm(prompt_kwargs)

    print(f"OpenAI: {openai_response}\n")
    print(f"Groq: {groq_response}\n")
    print(f"Anthropic: {anthropic_response}\n")
    print(f"Google GenAI: {google_gen_ai_response}\n")
    print(f"Ollama: {ollama_response}\n")
    # print(f"AWS Bedrock: {aws_bedrock_llm_response}\n")


if __name__ == "__main__":
    adal.setup_env()
    use_all_providers()
