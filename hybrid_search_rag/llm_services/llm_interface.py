# hybrid_search_rag/llm_services/llm_interface.py
# -*- coding: utf-8 -*-
"""
Interface for interacting with different Large Language Model APIs.
Handles requests sequentially through configured providers and their models
(e.g., Google Model 1 -> Google Model 2 -> Groq Model 1 -> Groq Model 2).
Includes fallback logic for rate limiting or other specified errors.
"""

import logging
import time
from typing import Optional, Union, Generator, List, Dict, Any, Tuple, Type, TypeAlias, TYPE_CHECKING # Added TYPE_CHECKING

from .. import config

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Uncomment for detailed debug logs from this module

# --- Define placeholder variables for SDKs and Exception types ---
genai: Any = None
# Groq: Any = None # Will be defined below
# Removed initial ChatCompletionMessageParam variable definition here

# --- Provider Specific Imports and Exception Handling ---
GOOGLE_AVAILABLE = False
BaseGoogleException = Exception
GoogleResourceExhausted: Type[BaseGoogleException]
GoogleAPICallError: Type[BaseGoogleException]
InvalidArgument: Type[BaseGoogleException]

try:
    import google.generativeai as genai_sdk # Use a public import alias
    from google.api_core.exceptions import ResourceExhausted as _RealGoogleResourceExhausted
    from google.api_core.exceptions import GoogleAPICallError as _RealGoogleAPICallError, InvalidArgument as _RealInvalidArgument
    
    genai = genai_sdk 
    GoogleResourceExhausted = _RealGoogleResourceExhausted
    GoogleAPICallError = _RealGoogleAPICallError
    InvalidArgument = _RealInvalidArgument
    GOOGLE_AVAILABLE = True
    logger.info("Google Generative AI SDK loaded successfully.")
except ImportError:
    logger.warning("Google Generative AI SDK not found (`pip install google-generativeai`). Google provider disabled.")
    class _DummyGoogleResourceExhausted(BaseGoogleException): pass
    class _DummyGoogleAPICallError(BaseGoogleException): pass
    class _DummyInvalidArgument(BaseGoogleException): pass
    GoogleResourceExhausted = _DummyGoogleResourceExhausted
    GoogleAPICallError = _DummyGoogleAPICallError
    InvalidArgument = _DummyInvalidArgument

# --- Groq Specific Imports and Exception Handling ---
GROQ_AVAILABLE = False
BaseGroqException = Exception 

# Runtime placeholders for Groq client and specific exceptions
Groq: Any = None
_GroqRateLimitErrorRuntime: Type[BaseGroqException] = BaseGroqException # Default to base
_GroqAPIErrorRuntime: Type[BaseGroqException] = BaseGroqException       # Default to base

if TYPE_CHECKING:
    # For type checker, assume groq is available and import specific types
    from groq import Groq as _GroqClientTyping
    # Corrected import path for ChatCompletionMessageParam
    from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam as _CCP_TypeTyping
    from groq.types.chat import ChatCompletion as _CC_TypeTyping
    from groq import RateLimitError as _RateLimitErrorTyping, APIError as _APIErrorTyping

    ChatCompletionMessageParamType: TypeAlias = _CCP_TypeTyping
    ChatCompletionType: TypeAlias = _CC_TypeTyping
    # Define TypeAliases for exception types for more precise type hints in `except` blocks if needed by the checker
    GroqRateLimitError_Hint: TypeAlias = _RateLimitErrorTyping
    GroqAPIError_Hint: TypeAlias = _APIErrorTyping
else:
    # For runtime, define TypeAliases with fallback types
    ChatCompletionMessageParamType: TypeAlias = Dict[str, str]
    ChatCompletionType: TypeAlias = Any
    # At runtime, GroqRateLimitError and GroqAPIError will be variables holding the actual exception classes.
    # No TypeAlias needed for them here in the runtime path for the `except` blocks themselves.

try:
    from groq import Groq as _GroqClientActual, RateLimitError as _RateLimitErrorActual, APIError as _APIErrorActual
    # If Groq is imported successfully, we expect the types used by the SDK to align with what
    # the type checker saw via TYPE_CHECKING. The TypeAliases will guide usage.
    # No need to redefine ChatCompletionMessageParamType and ChatCompletionType here for runtime,
    # as their definitions from the TYPE_CHECKING else block (or if block if type checker runs) will be used.

    Groq = _GroqClientActual  # Assign runtime client
    _GroqRateLimitErrorRuntime = _RateLimitErrorActual
    _GroqAPIErrorRuntime = _APIErrorActual
    GROQ_AVAILABLE = True
    logger.info("Groq SDK loaded successfully.")
except ImportError:
    logger.warning("Groq SDK not found (`pip install groq`). Groq provider disabled.")
    # GROQ_AVAILABLE is already False.
    # Groq, _GroqRateLimitError, _GroqAPIError already have their fallback values.
    # ChatCompletionMessageParamType and ChatCompletionType use their `else` definitions from TYPE_CHECKING.
    pass

# Assign the (potentially updated) exception classes to the names used in `except` blocks
GroqRateLimitError = _GroqRateLimitErrorRuntime
GroqAPIError = _GroqAPIErrorRuntime


def _get_llm_attempts() -> List[Dict[str, Any]]:
    attempts = []
    provider_details = {
        "google": {"sdk_available": GOOGLE_AVAILABLE, "models": config.LLM_GOOGLE_MODELS, "key": config.GOOGLE_API_KEY},
        "groq": {"sdk_available": GROQ_AVAILABLE, "models": config.LLM_GROQ_MODELS, "key": config.GROQ_API_KEY},
    }
    for provider_name in config.LLM_PROVIDER_ORDER:
        details = provider_details.get(provider_name)
        if not details: 
            logger.warning(f"Provider '{provider_name}' in LLM_PROVIDER_ORDER is not recognized.")
            continue
        if not details["sdk_available"]: 
            logger.info(f"SDK for provider '{provider_name}' not available. Skipping this provider.")
            continue
        if not details["key"]: 
            logger.warning(f"API key for provider '{provider_name}' not found. Skipping this provider.")
            continue
        if not details["models"]: 
            logger.warning(f"Model list for provider '{provider_name}' is empty. Skipping this provider.")
            continue
        for model_id in details["models"]:
            if model_id: attempts.append({"provider": provider_name, "model_id": model_id, "api_key": details["key"]})
    if not attempts: logger.error("No valid LLM attempts could be configured.")
    return attempts

def _prepare_google_gen_config(gen_args: Dict[str, Any]) -> Tuple[Optional[Any], Optional[List[Dict[str,str]]], Dict[str, Any]]:
    """Helper to prepare GenerationConfig and safety_settings for Google API."""
    current_call_gen_args = gen_args.copy()
    generation_config_params = {}
    for param in ['max_output_tokens', 'temperature', 'top_p', 'top_k', 'candidate_count', 'stop_sequences']:
        if param in current_call_gen_args:
            generation_config_params[param] = current_call_gen_args.pop(param)
    
    final_generation_config = None
    if generation_config_params and genai: 
        final_generation_config = genai.types.GenerationConfig(**generation_config_params)

    safety_settings_override = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    safety_settings_to_use = current_call_gen_args.pop('safety_settings', safety_settings_override)
    logger.debug(f"Using safety settings for Google API call: {safety_settings_to_use}")
    
    return final_generation_config, safety_settings_to_use, current_call_gen_args


# --- UNARY (Non-Streaming) Google API Call ---
def _call_google_api_unary(api_key: str, model_id: str, prompt: str, gen_args: Dict[str, Any]) -> Optional[str]:
    if not GOOGLE_AVAILABLE or genai is None:
        raise RuntimeError("Google Generative AI SDK is not available or not imported correctly for unary call.")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        logger.info(f"Sending UNARY request to Google API (Model: {model_id})...")

        generation_config, safety_settings, other_args = _prepare_google_gen_config(gen_args)

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            **other_args 
        )
        
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason_name = response.prompt_feedback.block_reason.name
            logger.warning(f"Google unary call blocked due to: {block_reason_name}")
            raise ValueError(f"Unary call blocked by Google safety filters: {block_reason_name}")

        if response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason not in [0, 1]: 
                    finish_reason_val = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else candidate.finish_reason
                    logger.warning(f"Unary response candidate finish_reason: {finish_reason_val}. Content may be incomplete or affected.")
                    if candidate.finish_reason == 2: 
                         logger.warning(f"Unary response hit MAX_TOKENS for model {model_id}.")
        
        if response.parts:
            full_text = "".join(part.text for part in response.parts if hasattr(part, 'text') and part.text)
            if full_text:
                logger.info(f"Google API unary call successful for model: {model_id}.")
                return full_text.strip()
        elif hasattr(response, 'text') and response.text: # Fallback
            logger.info(f"Google API unary call successful (using response.text) for model: {model_id}.")
            return response.text.strip()
        
        logger.warning(f"Google API unary call for model {model_id} returned no text content. Finish Reason (first candidate): {response.candidates[0].finish_reason.name if response.candidates and hasattr(response.candidates[0].finish_reason, 'name') else (response.candidates[0].finish_reason if response.candidates else 'N/A')}")
        return "" 

    except InvalidArgument as e: logger.error(f"Google API Invalid Argument (model: '{model_id}'): {e}", exc_info=True); raise 
    except GoogleResourceExhausted as e: logger.warning(f"Google API Resource Exhausted (model: {model_id}): {e}"); raise
    except GoogleAPICallError as e: logger.error(f"Google API call error (model: {model_id}): {e}", exc_info=True); raise
    except Exception as e: logger.error(f"Unexpected error during Google unary call (model: {model_id}): {e}", exc_info=True); raise
    return None


# --- STREAMING Google API Call ---
def _call_google_api_stream(api_key: str, model_id: str, prompt: str, gen_args: Dict[str, Any]) -> Generator[str, None, None]:
    if not GOOGLE_AVAILABLE or genai is None:
        raise RuntimeError("Google Generative AI SDK is not available or not imported correctly for stream call.")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        logger.info(f"Sending STREAMING request to Google API (Model: {model_id})...")
        
        generation_config, safety_settings, other_args = _prepare_google_gen_config(gen_args)

        response_stream = model.generate_content(
            prompt, stream=True, generation_config=generation_config,
            safety_settings=safety_settings, **other_args
        )
        text_yielded = False
        last_chunk = None
        for chunk in response_stream:
            last_chunk = chunk
            if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                 raise ValueError(f"Stream chunk blocked by Google safety filters: {chunk.prompt_feedback.block_reason.name}")
            if chunk.parts:
                for part in chunk.parts:
                    if hasattr(part, 'text') and part.text: yield part.text; text_yielded = True
            elif hasattr(chunk, 'text') and chunk.text: yield chunk.text; text_yielded = True
        
        if not text_yielded: logger.warning(f"Google stream for {model_id} yielded no text parts.")
        logger.info(f"Google API stream finished for model: {model_id}.")

    except InvalidArgument as e: logger.error(f"Google API Invalid Argument (stream, model: '{model_id}'): {e}", exc_info=True); raise 
    except GoogleResourceExhausted as e: logger.warning(f"Google API Resource Exhausted (stream, model: {model_id}): {e}"); raise
    except GoogleAPICallError as e: logger.error(f"Google API call error (stream, model: {model_id}): {e}", exc_info=True); raise
    except Exception as e: logger.error(f"Unexpected error Google stream (model: {model_id}): {e}", exc_info=True); raise

# --- UNARY (Non-Streaming) Groq API Call ---
def _call_groq_api_unary(api_key: str, model_id: str, prompt: str, gen_args: Dict[str, Any]) -> Optional[str]:
    if not GROQ_AVAILABLE or Groq is None:
        raise RuntimeError("Groq SDK is not available or not imported correctly for unary call.")
    try:
        client = Groq(api_key=api_key, timeout=config.LLM_API_TIMEOUT)
        logger.info(f"Sending UNARY request to Groq API (Model: {model_id})...")
        # Use the TypeAlias here
        messages: List[ChatCompletionMessageParamType] = [{"role": "user", "content": prompt}]
        
        groq_params = {key: gen_args[key] for key in ["temperature", "top_p"] if key in gen_args}
        if "max_output_tokens" in gen_args: groq_params["max_tokens"] = gen_args["max_output_tokens"]

        # Use the TypeAlias here
        completion: ChatCompletionType = client.chat.completions.create(
            messages=messages, model=model_id, stream=False, **groq_params 
        )
        
        if completion.choices and completion.choices[0].message:
            content = completion.choices[0].message.content
            if content is not None:
                logger.info(f"Groq API unary call successful for model: {model_id}.")
                return content.strip()
        
        logger.warning(f"Groq API unary call for model {model_id} returned no text content. Finish reason: {completion.choices[0].finish_reason if completion.choices else 'N/A'}")
        return "" 
        
    except GroqRateLimitError as e: logger.warning(f"Groq Rate Limit (unary, model: {model_id}): {e}"); raise
    except GroqAPIError as e: logger.error(f"Groq API Error (unary, model: {model_id}): {e}", exc_info=True); raise
    except Exception as e: logger.error(f"Unexpected Groq unary error (model: {model_id}): {e}", exc_info=True); raise
    return None


# --- STREAMING Groq API Call ---
def _call_groq_api_stream(api_key: str, model_id: str, prompt: str, gen_args: Dict[str, Any]) -> Generator[str, None, None]:
    if not GROQ_AVAILABLE or Groq is None:
        raise RuntimeError("Groq SDK is not available or not imported correctly for stream call.")
    try:
        client = Groq(api_key=api_key, timeout=config.LLM_API_TIMEOUT)
        logger.info(f"Sending STREAMING request to Groq API (Model: {model_id})...")
        # Use the TypeAlias here
        messages: List[ChatCompletionMessageParamType] = [{"role": "user", "content": prompt}]
        groq_params = {key: gen_args[key] for key in ["temperature", "top_p"] if key in gen_args}
        if "max_output_tokens" in gen_args: groq_params["max_tokens"] = gen_args["max_output_tokens"]

        stream = client.chat.completions.create(messages=messages, model=model_id, stream=True, **groq_params)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
        logger.info(f"Groq API stream finished successfully for model: {model_id}.")
    except GroqRateLimitError as e: logger.warning(f"Groq Rate Limit (stream, model: {model_id}): {e}"); raise
    except GroqAPIError as e: logger.error(f"Groq API Error (stream, model: {model_id}): {e}", exc_info=True); raise
    except Exception as e: logger.error(f"Unexpected Groq stream error (model: {model_id}): {e}", exc_info=True); raise


# --- Main LLM Dispatcher Functions ---

def get_llm_response_stream(prompt: str, generation_args: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
    llm_attempts = _get_llm_attempts()
    if not llm_attempts: yield "[Error: No LLM providers configured]"; return

    final_gen_args = {"max_output_tokens": config.LLM_MAX_NEW_TOKENS, "temperature": config.LLM_TEMPERATURE}
    if generation_args: final_gen_args.update(generation_args)
    logger.debug(f"Streaming call final_gen_args: {final_gen_args}")

    last_exception_message = "All attempts failed or yielded no content."
    for attempt in llm_attempts:
        provider, model_id, api_key = attempt["provider"], attempt["model_id"], attempt["api_key"]
        logger.info(f"Attempting STREAM: Provider='{provider}', Model='{model_id}'")
        try:
            gen_func = None
            if provider == "google": gen_func = _call_google_api_stream
            elif provider == "groq": gen_func = _call_groq_api_stream
            
            if not gen_func: logger.warning(f"No stream function for provider: {provider}"); continue

            content_yielded = False
            for chunk_text in gen_func(api_key, model_id, prompt, final_gen_args):
                yield chunk_text; content_yielded = True
            
            if content_yielded: logger.info(f"Stream success: {provider}/{model_id}"); return
            logger.warning(f"Stream from {provider}/{model_id} yielded no content.")
            last_exception_message = f"Empty stream from {provider}/{model_id}"

        except (GoogleResourceExhausted, GroqRateLimitError) as e: last_exception_message = f"Rate limit {provider}/{model_id}: {e}"; logger.warning(last_exception_message + ". Retrying..."); time.sleep(1); continue
        except ValueError as e: 
            if "blocked by safety filters" in str(e): last_exception_message = f"Safety block {provider}/{model_id}: {e}"; logger.warning(last_exception_message + ". Retrying..."); continue
            last_exception_message = f"ValueError {provider}/{model_id}: {e}"; logger.error(last_exception_message, exc_info=True); break
        except (GoogleAPICallError, GroqAPIError, InvalidArgument, RuntimeError) as e: last_exception_message = f"API/SDK Error {provider}/{model_id}: {e}"; logger.error(last_exception_message, exc_info=True); continue
        except Exception as e: last_exception_message = f"Unexpected Error {provider}/{model_id}: {e}"; logger.error(last_exception_message, exc_info=True); break
    
    yield f"[Error: All LLM stream attempts failed. Last: {last_exception_message}]"


def get_llm_response(prompt: str, generation_args: Optional[Dict[str, Any]] = None) -> Optional[str]:
    llm_attempts = _get_llm_attempts()
    if not llm_attempts: logger.error("No LLM attempts configured."); return "[Error: No LLM providers configured]"

    final_gen_args = {"max_output_tokens": config.LLM_MAX_NEW_TOKENS, "temperature": config.LLM_TEMPERATURE}
    if generation_args: final_gen_args.update(generation_args)
    logger.debug(f"Unary/Collected call final_gen_args: {final_gen_args}")

    last_exception_message: str = "All attempts failed."
    for attempt_info in llm_attempts:
        provider, model_id, api_key = attempt_info["provider"], attempt_info["model_id"], attempt_info["api_key"]
        logger.info(f"Attempting Unary/Collected: Provider='{provider}', Model='{model_id}'")
        try:
            response_text: Optional[str] = None
            if provider == "google" and GOOGLE_AVAILABLE:
                response_text = _call_google_api_unary(api_key, model_id, prompt, final_gen_args)
            elif provider == "groq" and GROQ_AVAILABLE:
                response_text = _call_groq_api_unary(api_key, model_id, prompt, final_gen_args) 
            else: logger.warning(f"Unsupported/unavailable provider for Unary/Collected: {provider}"); continue

            if response_text is not None: 
                logger.info(f"Unary/Collected call success: {provider}/{model_id} (Content length: {len(response_text)})")
                return response_text 
            
            last_exception_message = f"Call to {provider}/{model_id} returned None or failed internally."
            logger.warning(last_exception_message) 

        except (GoogleResourceExhausted, GroqRateLimitError) as e: last_exception_message = f"Rate limit {provider}/{model_id}: {e}"; logger.warning(last_exception_message + ". Retrying..."); time.sleep(1); continue
        except ValueError as e: 
            if "blocked by safety filters" in str(e): last_exception_message = f"Safety block {provider}/{model_id}: {e}"; logger.warning(last_exception_message + ". Retrying..."); continue
            last_exception_message = f"ValueError {provider}/{model_id}: {e}"; logger.error(last_exception_message, exc_info=True); break
        except (GoogleAPICallError, GroqAPIError, InvalidArgument, RuntimeError) as e: last_exception_message = f"API/SDK Error {provider}/{model_id}: {e}"; logger.error(last_exception_message, exc_info=True); continue
        except Exception as e: last_exception_message = f"Unexpected Error {provider}/{model_id}: {e}"; logger.error(last_exception_message, exc_info=True); break
            
    logger.error(f"All Unary/Collected LLM attempts failed. Last: {last_exception_message}")
    return None
