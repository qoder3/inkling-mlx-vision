"""Fair FIFO request queue and continuous SSE keepalive HTTP handler for single-GPU serving."""

from __future__ import annotations

import collections
import json
import threading
import time
import traceback
import uuid
from typing import Any, Callable

from runtime_mlx import server as _server


class FairQueue:
    """Strict FIFO queue for single-GPU generation with instant cancellation when a client disconnects."""

    def __init__(self):
        self._cond = threading.Condition()
        self._q: collections.deque[int] = collections.deque()
        self._next_id = 0

    def enter(self) -> tuple[int, int]:
        with self._cond:
            self._next_id += 1
            ticket = self._next_id
            self._q.append(ticket)
            return ticket, len(self._q)

    def wait_turn(self, ticket: int, gone_fn: Callable[[], bool]) -> None:
        with self._cond:
            while self._q and self._q[0] != ticket:
                if gone_fn():
                    try:
                        self._q.remove(ticket)
                    except ValueError:
                        pass
                    self._cond.notify_all()
                    raise _server.ClientGone()
                self._cond.wait(timeout=0.25)

    def leave(self, ticket: int) -> None:
        with self._cond:
            try:
                self._q.remove(ticket)
            except ValueError:
                pass
            self._cond.notify_all()


def make_queued_handler_factory(orig_make_handler, log_fn: Callable[[str], None]):
    """Return a `make_handler` replacement that queues concurrent requests instead of returning HTTP 503."""

    def _queued_make_handler(engine, model_id: str, *, api_key: str | None = None, loopback: bool = True):
        BaseHandler = orig_make_handler(engine, model_id, api_key=api_key, loopback=loopback)
        fifo = FairQueue()
        prep_lock = threading.Lock()

        class QueuedHandler(BaseHandler):
            def _error(self, status: int, message: str, etype: str = "invalid_request_error") -> None:
                log_fn(f"[http] {self.command} {self.path} -> {status} {etype}: {message}")
                super()._error(status, message, etype)

            def do_POST(self):
                if not self._authorized():
                    return
                path = self.path.split("?")[0].rstrip("/")
                utility = {"/apply-template": "render", "/tokenize": "tokenize"}.get(path)
                if utility is not None:
                    operation = getattr(engine, utility, None)
                    if operation is None:
                        self._error(404, "endpoint not found")
                        return
                elif path not in ("/v1/chat/completions", "/chat/completions"):
                    self._error(404, "endpoint not found")
                    return

                lengths = self.headers.get_all("Content-Length", [])
                if self.headers.get_all("Transfer-Encoding") or len(lengths) != 1:
                    self._error(400, "one Content-Length header is required")
                    return
                try:
                    length = int(lengths[0])
                except ValueError:
                    length = 0
                if length <= 0 or not lengths[0].isascii() or not lengths[0].isdigit():
                    self._error(400, "invalid Content-Length")
                    return
                if length > _server.MAX_REQUEST_BYTES:
                    self._error(413, "request body is too large")
                    return
                if self.headers.get_content_type() != "application/json":
                    self._error(415, "Content-Type must be application/json")
                    return
                try:
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("incomplete request body")
                    req = json.loads(raw)
                    if not isinstance(req, dict):
                        raise ValueError("request body must be a JSON object")
                except (ValueError, RecursionError):
                    self._error(400, "invalid JSON request body")
                    return
                except OSError:
                    return

                self._deadline.cancel()

                if utility is not None:
                    with prep_lock:
                        self._tokenizer_response(utility, operation, req)
                    return

                try:
                    if "stream" in req and type(req["stream"]) is not bool:
                        raise _server.RequestError(400, "stream must be a boolean")
                    if req.get("stream_options") is not None and not isinstance(req["stream_options"], dict):
                        raise _server.RequestError(400, "stream_options must be an object")
                    with prep_lock:
                        job = engine.prepare(req)
                except _server.RequestError as e:
                    self._error(e.status, e.message, e.etype)
                    return
                except Exception:
                    traceback.print_exc()
                    self._error(500, "server error", "server_error")
                    return

                if req.get("stream"):
                    self._queued_stream(req, job)
                else:
                    self._queued_complete(req, job)

            def _queued_complete(self, req: dict, job: dict) -> None:
                ticket, pos = fifo.enter()
                try:
                    if pos > 1:
                        log_fn(f"[queue] #{ticket} waiting (pos {pos}, prompt {len(job['ids'])} tok, non-stream)")
                    fifo.wait_turn(ticket, self._gone)
                    if pos > 1:
                        log_fn(f"[queue] #{ticket} acquired GPU")
                    self._complete(req, job)
                except _server.ClientGone:
                    log_fn(f"[queue] #{ticket} client disconnected while waiting")
                    self.close_connection = True
                finally:
                    fifo.leave(ticket)

            def _queued_stream(self, req: dict, job: dict) -> None:
                cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
                include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
                wlock = threading.Lock()
                done_evt = threading.Event()
                broken = threading.Event()
                last_write = [time.monotonic()]
                phase = ["queued"]
                n_tools = [0]

                def write(raw: bytes) -> None:
                    with wlock:
                        try:
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(raw), raw))
                            self.wfile.flush()
                            last_write[0] = time.monotonic()
                        except OSError:
                            broken.set()
                            raise _server.ClientGone()

                def chunk(delta, finish=None, usage=None) -> None:
                    obj: dict[str, Any] = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    if usage is not None:
                        obj["usage"] = usage
                    write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

                def emit(kind, payload) -> None:
                    phase[0] = "generating"
                    if kind == "tool":
                        chunk({"tool_calls": [dict(payload, index=n_tools[0])]})
                        n_tools[0] += 1
                    else:
                        chunk({"reasoning_content" if kind == "reasoning" else "content": payload})

                def keepalive() -> None:
                    while not done_evt.wait(1.0):
                        if time.monotonic() - last_write[0] >= 5.0:
                            try:
                                write(f": {phase[0]}\n\n".encode())
                            except _server.ClientGone:
                                return

                self.send_response(200)
                for k, v in (
                    ("Content-Type", "text/event-stream"),
                    ("Cache-Control", "no-cache"),
                    ("Transfer-Encoding", "chunked"),
                    ("Connection", "close"),
                ):
                    self.send_header(k, v)
                self.end_headers()

                ticket, pos = fifo.enter()
                ka = threading.Thread(target=keepalive, daemon=True)
                try:
                    chunk({"role": "assistant", "content": ""})
                    ka.start()
                    if pos > 1:
                        log_fn(f"[queue] #{ticket} waiting (pos {pos}, prompt {len(job['ids'])} tok, stream)")
                    fifo.wait_turn(ticket, lambda: broken.is_set() or self._gone())
                    if pos > 1:
                        log_fn(f"[queue] #{ticket} acquired GPU")
                    phase[0] = "prefilling"
                    usage, finish = engine.run(job, emit, lambda: broken.is_set() or self._gone())
                    done_evt.set()
                    ka.join()
                    chunk({}, finish)
                    if include_usage:
                        chunk(None, usage=usage)
                    write(b"data: [DONE]\n\n")
                    write(b"")
                except _server.ClientGone:
                    log_fn(f"[queue] #{ticket} client disconnected; stopped")
                    self.close_connection = True
                except Exception:
                    traceback.print_exc()
                    log_fn(f"[err] #{ticket} internal server error (see stderr traceback)")
                    try:
                        done_evt.set()
                        if ka.ident is not None:
                            ka.join()
                        write(
                            b"data: "
                            + json.dumps(
                                {"error": {"message": "server error", "type": "server_error", "code": 500}}
                            ).encode()
                            + b"\n\n"
                        )
                        write(b"data: [DONE]\n\n")
                        write(b"")
                    except _server.ClientGone:
                        pass
                finally:
                    done_evt.set()
                    if ka.ident is not None:
                        ka.join()
                    fifo.leave(ticket)

        return QueuedHandler

    return _queued_make_handler
