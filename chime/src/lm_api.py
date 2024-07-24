import hashlib
import os
import pickle
from typing import Any, Dict, Optional, Union

import anthropic
import openai

class LanguageModelAPI:
    def __init__(
        self,
        api_type: str,
        model: str,
        cache_dir: str,
        api_key: str,
        openai_system_content: Optional[str] = None,
    ):
        self.api_key = api_key
        self.api_type = api_type.lower()
        self.cache_dir = cache_dir
        self.model = model
        os.makedirs(cache_dir, exist_ok=True)
        if self.api_type == "claude":
            self.client = anthropic.Client(api_key=self.api_key)
            assert self.model in [
                "claude-v1",
                "claude-v1.2",
                "claude-v1.3",
                "claude-v1.4",
                "claude-2-instant",
                "claude-2"
            ]
        elif self.api_type == "openai":
            openai.api_key = self.api_key
            self.open_system_content = (
                openai_system_content or "You are a helpful assistant."
            )
            assert self.model in [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-0301",
                "gpt-4",
                "gpt-4-0613",
                "gpt-4-1106-preview",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-1106",
                "gpt-3.5-turbo-0125",
                "gpt-3.5-turbo-16k-0613"]
        else:
            raise ValueError("Invalid API type. Choose either 'claude' or 'openai'.")

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha1(prompt.encode()).hexdigest()

    def _cache_file_path(self, prompt: str) -> str:
        cache_key = self._cache_key(prompt)
        api_dir = os.path.join(self.cache_dir, self.api_type)
        model_dir = os.path.join(api_dir, self.model)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, f"{cache_key}.pkl")

    def _load_from_cache(self, prompt: str) -> Optional[Any]:
        cache_file = self._cache_file_path(prompt)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, prompt: str, response: Any) -> None:
        cache_file = self._cache_file_path(prompt)
        with open(cache_file, "wb") as f:
            pickle.dump(response, f)

    def chat(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self._get_response(prompt, max_tokens)
        if self.api_type == "claude":
            return response["completion"]
        elif self.api_type == "openai":
            return response.choices[0].message["content"].strip()

    def _get_response(self, prompt: str, max_tokens: int = 512, overwrite = False) -> Any:
        if overwrite:
            cached_response = None
        else:
            cached_response = self._load_from_cache(prompt)
        if cached_response is not None:
            return cached_response

        if self.api_type == "claude":
            response = self._get_response_anthropic(prompt, max_tokens)
        elif self.api_type == "openai":
            response = self._get_response_openai(prompt)

        self._save_to_cache(prompt, response)
        return response

    def _get_response_anthropic(self, prompt: str, max_tokens: int) -> Any:
        response = self.client.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.model,
            max_tokens_to_sample=max_tokens,
        )
        return response

    def _get_response_openai(self, prompt: str) -> Any:
        message_list = [
            {"role": "system", "content": self.open_system_content},
            {"role": "user", "content": prompt},
        ]

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=message_list,
        )
        return response