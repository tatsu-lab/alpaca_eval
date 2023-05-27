def get_decoder(name: str):
    """Get a decoder by name."""
    if name == "anthropic_completions":
        from .anthropic import anthropic_completions

        return anthropic_completions
    elif name == "openai_completions":
        from .openai import openai_completions

        return openai_completions
    else:
        raise ValueError(f"Unknown decoder: {name}")