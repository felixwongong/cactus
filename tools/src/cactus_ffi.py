import ctypes
import json
import platform
from pathlib import Path

TokenCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)

_DIR = Path(__file__).parent.parent.parent
if platform.system() == "Darwin":
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.dylib"
else:
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.so"

_lib = None
if _LIB_PATH.exists():
    _lib = ctypes.CDLL(str(_LIB_PATH))

    _lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    _lib.cactus_init.restype = ctypes.c_void_p

    _lib.cactus_complete.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_char_p, TokenCallback, ctypes.c_void_p
    ]
    _lib.cactus_complete.restype = ctypes.c_int

    _lib.cactus_transcribe.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_char_p, TokenCallback, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
    ]
    _lib.cactus_transcribe.restype = ctypes.c_int

    _lib.cactus_embed.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_bool
    ]
    _lib.cactus_embed.restype = ctypes.c_int

    _lib.cactus_image_embed.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.cactus_image_embed.restype = ctypes.c_int

    _lib.cactus_audio_embed.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.cactus_audio_embed.restype = ctypes.c_int

    _lib.cactus_reset.argtypes = [ctypes.c_void_p]
    _lib.cactus_reset.restype = None

    _lib.cactus_stop.argtypes = [ctypes.c_void_p]
    _lib.cactus_stop.restype = None

    _lib.cactus_destroy.argtypes = [ctypes.c_void_p]
    _lib.cactus_destroy.restype = None

    _lib.cactus_get_last_error.argtypes = []
    _lib.cactus_get_last_error.restype = ctypes.c_char_p

    _lib.cactus_set_telemetry_token.argtypes = [ctypes.c_char_p]
    _lib.cactus_set_telemetry_token.restype = None

    _lib.cactus_set_pro_key.argtypes = [ctypes.c_char_p]
    _lib.cactus_set_pro_key.restype = None

    _lib.cactus_tokenize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _lib.cactus_tokenize.restype = ctypes.c_int

    _lib.cactus_score_window.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.cactus_score_window.restype = ctypes.c_int

    _lib.cactus_rag_query.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t
    ]
    _lib.cactus_rag_query.restype = ctypes.c_int

    _lib.cactus_stream_transcribe_init.argtypes = [ctypes.c_void_p]
    _lib.cactus_stream_transcribe_init.restype = ctypes.c_void_p

    _lib.cactus_stream_transcribe_insert.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
    ]
    _lib.cactus_stream_transcribe_insert.restype = ctypes.c_int

    _lib.cactus_stream_transcribe_process.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p
    ]
    _lib.cactus_stream_transcribe_process.restype = ctypes.c_int

    _lib.cactus_stream_transcribe_finalize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
    ]
    _lib.cactus_stream_transcribe_finalize.restype = ctypes.c_int

    _lib.cactus_stream_transcribe_destroy.argtypes = [ctypes.c_void_p]
    _lib.cactus_stream_transcribe_destroy.restype = None


def cactus_init(model_path, context_size=2048, corpus_dir=None):
    return _lib.cactus_init(
        model_path.encode() if isinstance(model_path, str) else model_path,
        context_size,
        corpus_dir.encode() if corpus_dir else None
    )


def cactus_complete(
    model,
    messages,
    tools=None,
    temperature=None,
    top_p=None,
    top_k=None,
    max_tokens=None,
    stop_sequences=None,
    force_tools=False,
    callback=None
):
    if isinstance(messages, list):
        messages_json = json.dumps(messages)
    else:
        messages_json = messages

    tools_json = None
    if tools is not None:
        if isinstance(tools, list):
            tools_json = json.dumps(tools)
        else:
            tools_json = tools

    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if stop_sequences is not None:
        options["stop_sequences"] = stop_sequences
    if force_tools:
        options["force_tools"] = True

    options_json = json.dumps(options) if options else None

    buf = ctypes.create_string_buffer(65536)
    cb = TokenCallback(callback) if callback else TokenCallback()
    _lib.cactus_complete(
        model,
        messages_json.encode() if isinstance(messages_json, str) else messages_json,
        buf, len(buf),
        options_json.encode() if options_json else None,
        tools_json.encode() if tools_json else None,
        cb, None
    )
    return buf.value.decode("utf-8", errors="ignore")


def cactus_transcribe(model, audio_path, prompt="", callback=None):
    buf = ctypes.create_string_buffer(65536)
    cb = TokenCallback(callback) if callback else TokenCallback()
    _lib.cactus_transcribe(
        model,
        audio_path.encode() if isinstance(audio_path, str) else audio_path,
        prompt.encode() if isinstance(prompt, str) else prompt,
        buf, len(buf),
        None, cb, None, None, 0
    )
    return buf.value.decode()


def cactus_embed(model, text, normalize=False):
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_embed(
        model,
        text.encode() if isinstance(text, str) else text,
        buf, ctypes.sizeof(buf), ctypes.byref(dim), normalize
    )
    return list(buf[:dim.value])


def cactus_image_embed(model, image_path):
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_image_embed(
        model,
        image_path.encode() if isinstance(image_path, str) else image_path,
        buf, ctypes.sizeof(buf), ctypes.byref(dim)
    )
    return list(buf[:dim.value])


def cactus_audio_embed(model, audio_path):
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_audio_embed(
        model,
        audio_path.encode() if isinstance(audio_path, str) else audio_path,
        buf, ctypes.sizeof(buf), ctypes.byref(dim)
    )
    return list(buf[:dim.value])


def cactus_reset(model):
    _lib.cactus_reset(model)


def cactus_stop(model):
    _lib.cactus_stop(model)


def cactus_destroy(model):
    _lib.cactus_destroy(model)


def cactus_get_last_error():
    result = _lib.cactus_get_last_error()
    return result.decode() if result else None


def cactus_set_telemetry_token(token):
    _lib.cactus_set_telemetry_token(
        token.encode() if isinstance(token, str) else token
    )


def cactus_set_pro_key(pro_key):
    _lib.cactus_set_pro_key(
        pro_key.encode() if isinstance(pro_key, str) else pro_key
    )


def cactus_tokenize(model, text: str):
    needed = ctypes.c_size_t(0)
    rc = _lib.cactus_tokenize(
        model,
        text.encode("utf-8"),
        None,
        0,
        ctypes.byref(needed),
    )
    if rc != 0:
        raise RuntimeError(f"cactus_tokenize length query failed rc={rc}")

    n = needed.value
    arr = (ctypes.c_uint32 * n)()

    rc = _lib.cactus_tokenize(
        model,
        text.encode("utf-8"),
        arr,
        n,
        ctypes.byref(needed),
    )
    if rc != 0:
        raise RuntimeError(f"cactus_tokenize fetch failed rc={rc}")

    return [arr[i] for i in range(n)]


def cactus_score_window(model, tokens, start, end, context):
    buf = ctypes.create_string_buffer(4096)
    n = len(tokens)
    arr = (ctypes.c_uint32 * n)(*tokens)

    _lib.cactus_score_window(
        model,
        arr,
        n,
        start,
        end,
        context,
        buf,
        len(buf),
    )
    return json.loads(buf.value.decode("utf-8", errors="ignore"))


def cactus_rag_query(model, query, top_k=5):
    buf = ctypes.create_string_buffer(65536)
    result = _lib.cactus_rag_query(
        model,
        query.encode() if isinstance(query, str) else query,
        buf, len(buf), top_k
    )
    if result != 0:
        return []
    return json.loads(buf.value.decode("utf-8", errors="ignore"))


def cactus_stream_transcribe_init(model):
    return _lib.cactus_stream_transcribe_init(model)


def cactus_stream_transcribe_insert(stream, pcm_data):
    if isinstance(pcm_data, bytes):
        arr = (ctypes.c_uint8 * len(pcm_data)).from_buffer_copy(pcm_data)
    else:
        arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
    return _lib.cactus_stream_transcribe_insert(stream, arr, len(arr))


def cactus_stream_transcribe_process(stream, options=None):
    buf = ctypes.create_string_buffer(65536)
    _lib.cactus_stream_transcribe_process(
        stream, buf, len(buf),
        options.encode() if options else None
    )
    return buf.value.decode("utf-8", errors="ignore")


def cactus_stream_transcribe_finalize(stream):
    buf = ctypes.create_string_buffer(65536)
    _lib.cactus_stream_transcribe_finalize(stream, buf, len(buf))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_stream_transcribe_destroy(stream):
    _lib.cactus_stream_transcribe_destroy(stream)
