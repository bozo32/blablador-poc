#!/usr/bin/env python3
import argparse
import sys
import requests


def get_models(api_key: str, base_url: str):
    """Fetches the list of available models."""
    url = f"{base_url.rstrip('/')}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return [m["id"] for m in payload.get("data", [])]


def test_model(api_key: str, base_url: str, model_id: str):
    """Sends a tiny completion request to verify the model is working."""
    url = f"{base_url.rstrip('/')}/v1/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {"model": model_id, "prompt": "Test", "max_tokens": 1}
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    # capture status and truncated response for brevity
    return resp.status_code, resp.text[:200].replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(
        description="List and test Blablador completion models"
    )
    parser.add_argument("-k", "--api_key", required=True, help="Your Blablador API key")
    parser.add_argument(
        "-b",
        "--base_url",
        default="https://api.helmholtz-blablador.fz-juelich.de",
        help="Blablador Base URL",
    )
    args = parser.parse_args()

    # 1) List
    try:
        models = get_models(args.api_key, args.base_url)
    except Exception as e:
        print(f"Error fetching model list: {e}", file=sys.stderr)
        sys.exit(1)

    print("Available models:")
    for m in models:
        print(f"  • {m}")

    # 2) Test each
    print("\nTesting each model with a tiny completion request:")
    for m in models:
        try:
            status, snippet = test_model(args.api_key, args.base_url, m)
            print(f"{m:<30} → {status}   {snippet}")
        except Exception as e:
            print(f"{m:<30} → ERROR: {e}")


if __name__ == "__main__":
    main()
