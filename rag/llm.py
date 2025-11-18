"""LLM adapter module for RAG Assistant.

Handles communication with Ollama for text generation.
"""

import logging
import time
from typing import Optional

import httpx

from rag.config import Config

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Adapter for Ollama LLM API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 120.0,
    ):
        """Initialize the Ollama LLM adapter.

        Args:
            base_url: Ollama API base URL.
            model: Model name to use.
            temperature: Sampling temperature (0-1).
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model = model or Config.OLLAMA_MODEL
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        self.top_p = top_p if top_p is not None else Config.LLM_TOP_P
        self.max_tokens = max_tokens or Config.LLM_MAX_TOKENS
        self.timeout = timeout

        # Remove trailing slash from base URL
        self.base_url = self.base_url.rstrip("/")

        logger.info(
            f"Initialized Ollama LLM: model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            Generated text response.

        Raises:
            LLMError: If generation fails.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        # Build messages for chat endpoint
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "top_p": self.top_p,
                "num_predict": tokens,
            },
        }

        url = f"{self.base_url}/api/chat"

        logger.info(
            f"Generating response: model={self.model}, "
            f"temperature={temp}, max_tokens={tokens}"
        )
        logger.debug(f"Prompt length: {len(prompt)} chars")

        start_time = time.time()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            raise LLMError(
                f"Cannot connect to Ollama. Is it running? "
                f"Check: curl {self.base_url}/api/tags"
            ) from e

        except httpx.TimeoutException as e:
            elapsed = time.time() - start_time
            logger.error(f"Ollama request timed out after {elapsed:.1f}s: {e}")
            raise LLMError(
                f"Request timed out after {self.timeout}s. "
                f"Try reducing max_tokens or using a smaller model."
            ) from e

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 404:
                raise LLMError(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                ) from e
            raise LLMError(f"Ollama API error: {e.response.text}") from e

        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            raise LLMError(f"Generation failed: {e}") from e

        elapsed = time.time() - start_time

        # Extract response text
        try:
            response_text = result["message"]["content"]
        except KeyError as e:
            logger.error(f"Unexpected response format: {result}")
            raise LLMError(f"Invalid response format from Ollama: {e}") from e

        # Log performance metrics
        tokens_generated = result.get("eval_count", 0)
        tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0

        logger.info(
            f"Generated {len(response_text)} chars in {elapsed:.2f}s "
            f"({tokens_per_second:.1f} tokens/s)"
        )

        return response_text

    def check_health(self) -> dict:
        """Check if Ollama is running and model is available.

        Returns:
            Health status dictionary.
        """
        result = {
            "healthy": False,
            "ollama_running": False,
            "model_available": False,
            "model": self.model,
            "base_url": self.base_url,
        }

        try:
            with httpx.Client(timeout=5.0) as client:
                # Check if Ollama is running
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                result["ollama_running"] = True

                # Check if model is available
                models_data = response.json()
                available_models = [m["name"] for m in models_data.get("models", [])]

                # Check both exact match and without tag
                model_base = self.model.split(":")[0]
                result["model_available"] = (
                    self.model in available_models
                    or any(m.startswith(model_base) for m in available_models)
                )
                result["available_models"] = available_models

                result["healthy"] = result["ollama_running"] and result["model_available"]

        except httpx.ConnectError:
            result["error"] = f"Cannot connect to Ollama at {self.base_url}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def get_config(self) -> dict:
        """Get current LLM configuration.

        Returns:
            Configuration dictionary.
        """
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }


class LLMError(Exception):
    """Custom exception for LLM-related errors."""

    pass


def create_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> OllamaLLM:
    """Convenience function to create an LLM instance.

    Args:
        model: Model name to use.
        temperature: Sampling temperature.

    Returns:
        Configured OllamaLLM instance.
    """
    return OllamaLLM(model=model, temperature=temperature)


if __name__ == "__main__":
    # Configure logging for standalone run
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create LLM and check health
    llm = create_llm()

    print("Checking Ollama health...")
    health = llm.check_health()

    if health["healthy"]:
        print(f"Ollama is healthy. Model '{llm.model}' is available.")

        # Test generation
        print("\nTesting generation...")
        response = llm.generate(
            prompt="What is 2 + 2? Answer in one word.",
            system_prompt="You are a helpful assistant. Be concise.",
        )
        print(f"Response: {response}")
    else:
        print(f"Ollama health check failed:")
        if not health["ollama_running"]:
            print(f"  - Ollama is not running at {health['base_url']}")
        elif not health["model_available"]:
            print(f"  - Model '{health['model']}' is not available")
            print(f"  - Available models: {health.get('available_models', [])}")
        if "error" in health:
            print(f"  - Error: {health['error']}")
