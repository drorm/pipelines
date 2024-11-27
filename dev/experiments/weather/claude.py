"""
Helper module for Claude API interactions
"""

import requests


def update_headers(api_key: str) -> dict:
    """Update the headers for Anthropic API requests"""
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "accept": "application/json",
    }


def anthropic_completion(
    messages: list,
    api_key: str,
    model: str = "claude-3-haiku-20240307",
    temperature: float = 0.7,
    max_tokens: int = 150,
) -> str:
    """Get completions from Anthropic API"""
    headers = update_headers(api_key)
    anthropic_url = "https://api.anthropic.com/v1/messages"

    payload = {
        "model": model,
        "messages": [
            {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
            for msg in messages
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(anthropic_url, headers=headers, json=payload)

        if response.status_code == 200:
            res = response.json()
            return (
                res["content"][0]["text"] if "content" in res and res["content"] else ""
            )
        else:
            print(f"Error response body: {response.text}")
            raise Exception(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {str(e)}")
        raise
