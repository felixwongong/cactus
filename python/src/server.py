import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .cactus import (
    cactus_init,
    cactus_destroy,
    cactus_reset,
    cactus_complete,
    cactus_embed,
    cactus_get_last_error,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

app = FastAPI(title="Cactus", version="0.1.0")


class ModelManager:
    def __init__(self, context_length: Optional[int] = None):
        self.current_model_id: Optional[str] = None
        self.current_handle = None
        self.lock = asyncio.Lock()
        self.context_length = context_length

    def _load(self, model_id: str):
        model_path = WEIGHTS_DIR / model_id
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in weights/")
        handle = cactus_init(str(model_path), context_length=self.context_length or 0)
        if not handle:
            err = cactus_get_last_error() or "unknown error"
            raise HTTPException(status_code=500, detail=f"Failed to load model: {err}")
        return handle

    async def get(self, model_id: str):
        async with self.lock:
            if self.current_model_id == model_id and self.current_handle:
                return self.current_handle
            if self.current_handle:
                cactus_destroy(self.current_handle)
                self.current_handle = None
                self.current_model_id = None
            handle = await asyncio.get_event_loop().run_in_executor(None, self._load, model_id)
            self.current_handle = handle
            self.current_model_id = model_id
            return handle

    def shutdown(self):
        if self.current_handle:
            cactus_destroy(self.current_handle)
            self.current_handle = None
            self.current_model_id = None


manager = ModelManager()


EMBEDDING_MODEL_TYPES = {"bert"}


def _read_model_type(model_dir: Path) -> str:
    config = model_dir / "config.txt"
    if not config.exists():
        return ""
    for line in config.read_text().splitlines():
        if line.startswith("model_type="):
            return line.split("=", 1)[1].strip()
    return ""


def discover_models():
    models = []
    if not WEIGHTS_DIR.exists():
        return models
    for entry in sorted(WEIGHTS_DIR.iterdir()):
        if entry.is_dir() and (entry / "config.txt").exists():
            stat = entry.stat()
            models.append({
                "id": entry.name,
                "object": "model",
                "created": int(stat.st_mtime),
                "owned_by": "cactus",
            })
    return models


# --- Request / Response Models ---

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class ToolChoiceFunction(BaseModel):
    name: str

class ToolChoiceObject(BaseModel):
    type: str = "function"
    function: ToolChoiceFunction

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    stream: bool = False
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[str | ToolChoiceObject] = None

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Optional[str] = None


# --- Helpers ---

def translate_tools(tools: Optional[list[Tool]], tool_choice) -> tuple[Optional[list[dict]], bool]:
    if tools is None:
        return None, False

    if isinstance(tool_choice, str) and tool_choice == "none":
        return None, False

    force_tools = False
    cactus_tools = None

    if isinstance(tool_choice, ToolChoiceObject):
        target_name = tool_choice.function.name
        for t in tools:
            if t.function.name == target_name:
                cactus_tools = [{"name": t.function.name, "description": t.function.description or "", "parameters": t.function.parameters or {}}]
                break
        force_tools = True
    else:
        cactus_tools = [
            {"name": t.function.name, "description": t.function.description or "", "parameters": t.function.parameters or {}}
            for t in tools
        ]
        if isinstance(tool_choice, str) and tool_choice == "required":
            force_tools = True

    return cactus_tools, force_tools


def make_tool_calls(function_calls: list) -> list[dict]:
    out = []
    for fc in function_calls:
        args = fc.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        out.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": fc.get("name", ""),
                "arguments": args_str,
            },
        })
    return out


def build_chat_response(response_json: dict, model_id: str, request_id: str) -> dict:
    function_calls = response_json.get("function_calls", [])
    has_tool_calls = len(function_calls) > 0
    text = response_json.get("response") or ""

    tool_calls = make_tool_calls(function_calls) if has_tool_calls else None

    if has_tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    message: dict = {"role": "assistant", "content": text if not has_tool_calls else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    prefill = response_json.get("prefill_tokens", 0)
    decode = response_json.get("decode_tokens", 0)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prefill,
            "completion_tokens": decode,
            "total_tokens": prefill + decode,
        },
    }


# --- Endpoints ---

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": discover_models()}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    handle = await manager.get(req.model)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    cactus_tools, force_tools = translate_tools(req.tools, req.tool_choice)

    kwargs: dict = {}
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    if req.top_k is not None:
        kwargs["top_k"] = req.top_k
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens
    if req.stop:
        kwargs["stop_sequences"] = req.stop
    if force_tools:
        kwargs["force_tools"] = True
    if cactus_tools:
        kwargs["tools"] = cactus_tools

    if req.stream:
        return StreamingResponse(
            _stream_completion(handle, messages, kwargs, request_id, req.model),
            media_type="text/event-stream",
        )

    async with manager.lock:
        cactus_reset(handle)
        raw = await asyncio.get_event_loop().run_in_executor(
            None, lambda: cactus_complete(handle, messages, **kwargs)
        )

    response_json = json.loads(raw)
    if not response_json.get("success", False):
        raise HTTPException(status_code=500, detail=response_json.get("error", "completion failed"))

    return build_chat_response(response_json, req.model, request_id)


async def _stream_completion(handle, messages, kwargs, request_id, model_id):
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_token(token_bytes, token_id, user_data):
        token_str = token_bytes.decode("utf-8", errors="ignore") if token_bytes else ""
        loop.call_soon_threadsafe(queue.put_nowait, token_str)

    async def run_inference():
        async with manager.lock:
            cactus_reset(handle)
            raw = await loop.run_in_executor(
                None,
                lambda: cactus_complete(handle, messages, callback=on_token, **kwargs),
            )
        loop.call_soon_threadsafe(queue.put_nowait, None)
        return raw

    task = asyncio.create_task(run_inference())

    first_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    while True:
        token = await queue.get()
        if token is None:
            break
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    raw = await task
    response_json = json.loads(raw)

    function_calls = response_json.get("function_calls", [])
    has_tool_calls = len(function_calls) > 0

    if has_tool_calls:
        finish_reason = "tool_calls"
        tool_calls = make_tool_calls(function_calls)
        tool_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(tool_chunk)}\n\n"
    else:
        finish_reason = "stop"

    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }

    prefill = response_json.get("prefill_tokens", 0)
    decode = response_json.get("decode_tokens", 0)
    final_chunk["usage"] = {
        "prompt_tokens": prefill,
        "completion_tokens": decode,
        "total_tokens": prefill + decode,
    }

    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    model_type = _read_model_type(WEIGHTS_DIR / req.model)
    if model_type not in EMBEDDING_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' is not an embedding model. Use an embedding model like nomic-embed-text-v2-moe.",
        )
    handle = await manager.get(req.model)

    inputs = req.input if isinstance(req.input, list) else [req.input]
    data = []

    async with manager.lock:
        for i, text in enumerate(inputs):
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, lambda t=text: cactus_embed(handle, t)
            )
            data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i,
            })

    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.on_event("shutdown")
def on_shutdown():
    manager.shutdown()


def create_app(context_length: Optional[int] = None) -> FastAPI:
    manager.context_length = context_length
    return app
