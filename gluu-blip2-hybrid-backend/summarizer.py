# summarizer.py
from openai import OpenAI
import os

def summarize_result(product_type: str, material: str, tags: list, openai_key: str=None) -> str:
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OpenAI API key not provided.")

    client = OpenAI(api_key=openai_key)
    tag_string = ", ".join(tags) if tags else "no visible flaws"
    prompt = (
        f"Write a 50-word condition summary for a {material} {product_type} based on these observations: {tag_string}. "
        "Describe flaws naturally if present, or highlight good condition otherwise. Avoid exaggeration."
    )

    messages = [
        {"role": "system", "content": "You are a concise product-condition summarizer."},
        {"role": "user", "content": prompt}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=120
    )
    # OpenAI python returns as choices[0].message.content in current SDK
    return completion.choices[0].message.content.strip()
