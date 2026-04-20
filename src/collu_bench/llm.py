from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .config import LLMConfig
from .prompt import PromptPayload

"""
LLM client implementations and generation postprocessing.

The Racket-specific helpers in this module are important because generative
models frequently emit partially formatted answers for Lisp-family languages:
Markdown fences, explanatory preambles, visible tokenizer markers, or code with
unbalanced delimiters.  The cleanup path attempts to recover an executable
`#lang racket` module without changing the intended program more than
necessary.
"""


@dataclass
class LLMGeneration:
    text: str
    tokens: List[str]
    token_logprobs: List[Dict[str, Any]]
    raw_response: Dict[str, Any]


class BaseLLMClient:
    """Interface for LLM providers."""

    def generate(self, prompt: PromptPayload, request_id: str) -> LLMGeneration:
        raise NotImplementedError


class LocalHFClient(BaseLLMClient):
    """Local Hugging Face client that performs inference on-device."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_id = config.local_model_path or config.model
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(config.dtype, self.device)
        self._prepare_environment()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        self._pad_added = False
        self._ensure_pad_token()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "auto" else None,
            trust_remote_code=True,
        )
        if self._pad_added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()
        self.model_device = next(self.model.parameters()).device

    def _prepare_environment(self) -> None:
        for key, value in self.config.environment.items():
            if value.startswith("$"):
                env_name = value[1:]
                resolved = os.getenv(env_name)
                if resolved is None:
                    raise EnvironmentError(
                        f"Missing value for {env_name} referenced in LLM environment"
                    )
                os.environ[key] = resolved
            else:
                os.environ[key] = value

    def _ensure_pad_token(self) -> None:
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self._pad_added = True

    def generate(self, prompt: PromptPayload, request_id: str) -> LLMGeneration:
        prompt_text, inputs = _render_prompt_and_tokenize(self.tokenizer, prompt, self.model_device)
        do_sample = self.config.temperature > 0

        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_tokens,
            temperature=max(self.config.temperature, 1e-5) if do_sample else 1.0,
            top_p=self.config.top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        full_sequence = outputs.sequences[0]
        generated_ids = full_sequence[prompt_length:]

        raw_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        tokens, token_logprobs = _collect_token_info(
            tokenizer=self.tokenizer,
            scores=outputs.scores,
            generated_ids=generated_ids,
            top_k=self.config.logprobs,
        )

        cleaned_text = _postprocess_generated_text(
            raw_text=raw_text,
            token_strings=tokens,
            prompt=prompt,
        )

        raw_response = {
            "prompt": prompt_text,
            "prompt_length": prompt_length,
            "generated_token_ids": generated_ids.tolist(),
            "sequence_token_ids": full_sequence.tolist(),
            "request_id": request_id,
            "raw_text_preview": raw_text[:1000],
        }

        return LLMGeneration(
            text=cleaned_text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            raw_response=raw_response,
        )


def _render_prompt_and_tokenize(tokenizer, payload: PromptPayload, device) -> tuple[str, Dict[str, torch.Tensor]]:
    """
    Render prompt text and tokenize it.

    For chat prompts, prefer tokenizer chat templates if available.
    Otherwise fall back to the legacy 'Role: content' rendering.
    """
    if payload.mode == "chat":
        messages = payload.content if isinstance(payload.content, list) else []

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = []
            for message in messages:
                role = message.get("role", "user").capitalize()
                rendered.append(f"{role}: {message.get('content', '')}".strip())
            rendered.append("Assistant:")
            prompt_text = "\n".join(rendered).strip()
    else:
        prompt_text = str(payload.content)

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return prompt_text, inputs


def _resolve_dtype(configured: Optional[str], device: str) -> torch.dtype:
    alias = (configured or "").lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if alias in mapping:
        return mapping[alias]
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _collect_token_info(
    tokenizer,
    scores: Sequence[torch.Tensor],
    generated_ids: torch.Tensor,
    top_k: int,
) -> tuple[List[str], List[Dict[str, Any]]]:
    if generated_ids.numel() == 0:
        return [], []

    raw_token_text = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
    token_text = [_clean_token_for_logging(tok) for tok in raw_token_text]

    token_logprobs: List[Dict[str, Any]] = []
    for step, token_id_tensor in enumerate(generated_ids):
        token_id = int(token_id_tensor.item())
        score = scores[step]
        logprob_tensor = torch.log_softmax(score[0], dim=-1)
        token_logprob = float(logprob_tensor[token_id].item())
        decoded = token_text[step]

        if top_k > 0:
            values, indices = torch.topk(
                logprob_tensor,
                k=min(top_k, logprob_tensor.shape[-1]),
            )
            top_entries = [
                {
                    "decoded_token": _clean_token_for_logging(
                        tokenizer.decode(
                            [idx.item()],
                            clean_up_tokenization_spaces=False,
                        )
                    ),
                    "token_id": int(idx.item()),
                    "logprob": float(val.item()),
                }
                for val, idx in zip(values, indices)
            ]
        else:
            top_entries = []

        token_logprobs.append(
            {
                "decoded_token": decoded,
                "token_id": token_id,
                "logprob": token_logprob,
                "top_logprobs": top_entries,
            }
        )

    return token_text, token_logprobs

def _postprocess_generated_text(
    raw_text: str,
    token_strings: List[str],
    prompt: PromptPayload,
) -> str:
    """
    Generic postprocessing with an extra cleanup path for Racket prompts.

    Racket is singled out here because parenthesized syntax is especially
    sensitive to even small formatting artifacts introduced by chat-oriented
    models.
    """
    text = raw_text.strip()

    if _looks_like_racket_prompt(prompt):
        return _cleanup_racket_output(text, token_strings)

    return text


def _looks_like_racket_prompt(prompt: PromptPayload) -> bool:
    """
    Heuristic detection of Racket generation prompts.
    """
    payload_text = ""
    if prompt.mode == "chat":
        if isinstance(prompt.content, list):
            payload_text = "\n".join(str(msg.get("content", "")) for msg in prompt.content)
    else:
        payload_text = str(prompt.content)

    lowered = payload_text.lower()
    return "racket" in lowered or "#lang racket" in lowered


def _normalize_visible_token_markers(text: str) -> str:
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    text = text.replace("ĉ", "\n")
    text = text.replace("▁", " ")
    text = text.replace("<0x0A>", "\n")
    return text

def _clean_token_for_logging(token: str) -> str:
    """
    Make tokenizer tokens more readable in CSV/log artifacts.

    We keep the raw generation behavior unchanged and only clean the token text
    that is logged in `tokens` / `token_logprobs`.
    """
    if not token:
        return token

    cleaned = token
    cleaned = cleaned.replace("Ġ", " ")
    cleaned = cleaned.replace("Ċ", "\n")
    cleaned = cleaned.replace("ĉ", "\n")
    cleaned = cleaned.replace("▁", " ")
    cleaned = cleaned.replace("<0x0A>", "\n")

    special_map = {
        "<|EOT|>": "<EOT>",
    }
    cleaned = special_map.get(cleaned, cleaned)

    return cleaned

def _detokenize_fallback_from_token_strings(tokens: List[str]) -> str:
    if not tokens:
        return ""

    text = "".join(tokens)
    text = _normalize_visible_token_markers(text)

    special_tokens_to_remove = [
        "<｜begin▁of▁sentence｜>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<s>",
        "</s>",
        "<pad>",
        "<|EOT|>",
    ]
    for token in special_tokens_to_remove:
        text = text.replace(token, "")

    return text


def _isolate_racket_code_region(text: str) -> str:
    lang_index = text.find("#lang racket")
    define_index = text.find("(define")

    if lang_index != -1:
        return text[lang_index:]
    if define_index != -1:
        return "#lang racket\n\n" + text[define_index:]
    return text


def _remove_known_prose_prefixes(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines: List[str] = []

    patterns = [
        r"^A:\s*Here is.*?$",
        r"^Here is.*?$",
        r"^Sure,.*?$",
        r"^Certainly,.*?$",
    ]

    for line in lines:
        stripped = line.strip()
        matched = False
        for pattern in patterns:
            if re.match(pattern, stripped, flags=re.IGNORECASE):
                matched = True
                break
        if not matched:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _truncate_after_last_balanced_form(text: str) -> str:
    last_balanced_end = -1
    depth = 0
    in_string = False
    escaped = False

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                last_balanced_end = idx + 1

    if last_balanced_end != -1:
        return text[:last_balanced_end]

    return text


def _balance_racket_brackets(text: str) -> str:
    stack: List[str] = []
    in_string = False
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {")", "]", "}"}

    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in closing:
            if stack and stack[-1] == ch:
                stack.pop()

    if stack:
        text += "".join(reversed(stack))

    return text


def _cleanup_racket_output(raw_output: str, token_strings: List[str]) -> str:
    """
    Convert a raw model response into a minimal executable Racket module.

    The cleanup order reflects the error profile observed in practice for
    Racket generation: remove chat prose, isolate the code region, keep only
    balanced forms, and finally restore a `#lang racket` header if the model
    omitted it.
    """
    decoded = raw_output.replace("\r\n", "\n").replace("\r", "\n").strip()
    decoded = _normalize_visible_token_markers(decoded)

    if "Ġ" in decoded or "Ċ" in decoded or "▁" in decoded:
        decoded = _detokenize_fallback_from_token_strings(token_strings).strip()

    fenced_match = re.search(
        r"```(?:racket|scheme|lisp)?\s*(.*?)```",
        decoded,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_match:
        decoded = fenced_match.group(1).strip()

    decoded = _remove_known_prose_prefixes(decoded)
    decoded = _isolate_racket_code_region(decoded)
    decoded = _truncate_after_last_balanced_form(decoded).strip()
    decoded = _balance_racket_brackets(decoded).strip()

    if not decoded.startswith("#lang racket"):
        decoded = "#lang racket\n\n" + decoded

    decoded = decoded.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not decoded.endswith("\n"):
        decoded += "\n"

    return decoded
