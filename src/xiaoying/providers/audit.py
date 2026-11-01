"""Public model request/response records exclude private reasoning and credentials."""

PRIVATE_FIELDS = {
    "reasoning_content", "reasoning", "chain_of_thought", "api_key", "authorization",
    "password", "cookie", "headers", "deepseek_api_key",
}


def public_model_payload(value):
    if isinstance(value, dict):
        return {
            key: public_model_payload(item) for key, item in value.items()
            if key.lower() not in PRIVATE_FIELDS
        }
    if isinstance(value, list):
        return [public_model_payload(item) for item in value]
    return value
