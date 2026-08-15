"""HTTP server for the Quantum Professor web UI."""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Importable two ways on purpose: as a package (`python -m <pkg>.web_server`,
# how it runs locally) and as a flat script (`python web_server.py`, how the
# deployment runs it, since the checkout directory is the package itself).
try:
    from .library import TOPICS, BOOKS, learning_path, topics_at_level, books_at_level
    from .physics import solve, SOLVERS
    from .problems import problems_for_topic
    from .serialize import topic_to_dict, book_to_dict, problem_to_dict
except ImportError:                                    # flat-script execution
    from library import TOPICS, BOOKS, learning_path, topics_at_level, books_at_level
    from physics import solve, SOLVERS
    from problems import problems_for_topic
    from serialize import topic_to_dict, book_to_dict, problem_to_dict

WEB_ROOT = Path(__file__).parent / "web"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".woff2": "font/woff2",          # vendored KaTeX fonts
}


class QuantumHandler(BaseHTTPRequestHandler):
    server_version = "QuantumProfUI/0.1"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parsed.query

        if path == "/api/topics":
            self._handle_topics(query)
        elif path == "/api/topic":
            self._handle_topic(query)
        elif path == "/api/books":
            self._handle_books(query)
        elif path == "/api/solve":
            self._handle_solve(query)
        elif path == "/api/path":
            self._handle_path(query)
        elif path == "/api/problems":
            self._handle_problems(query)
        elif path == "/api/ask":
            self._handle_ask(query)
        elif path == "/api/math":
            self._handle_math(query)
        elif path == "/api/ode":
            self._handle_ode(query)
        elif path == "/api/schrodinger":
            self._handle_schrodinger(query)
        elif path == "/api/mathlab":
            self._handle_mathlab_info(query)
        else:
            self._serve_static(path)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _handle_topics(self, query: str) -> None:
        params = parse_qs(query)
        level = params.get("level", [None])[0]
        if level:
            ts = topics_at_level(level)
        else:
            ts = list(TOPICS.values())
        self._send_json({"topics": [topic_to_dict(t) for t in ts]})

    def _handle_topic(self, query: str) -> None:
        params = parse_qs(query)
        topic_id = params.get("id", [""])[0].strip()
        if not topic_id or topic_id not in TOPICS:
            self._send_json({"error": f"Unknown topic: {topic_id!r}"}, status=404)
            return
        t = TOPICS[topic_id]
        d = topic_to_dict(t)
        d["problems"] = [problem_to_dict(p) for p in problems_for_topic(topic_id)]
        self._send_json(d)

    def _handle_books(self, query: str) -> None:
        params = parse_qs(query)
        level = params.get("level", [None])[0]
        if level:
            bs = books_at_level(level)
        else:
            bs = list(BOOKS.values())
        self._send_json({"books": [book_to_dict(b) for b in bs]})

    def _handle_solve(self, query: str) -> None:
        params = parse_qs(query)
        topic_id = params.get("topic", [""])[0].strip()
        if not topic_id:
            self._send_json({"error": "topic parameter required"}, status=400)
            return
        kwargs: dict = {}
        for k, vs in params.items():
            if k == "topic":
                continue
            v = vs[0]
            try:
                kwargs[k] = int(v)
            except ValueError:
                try:
                    kwargs[k] = float(v)
                except ValueError:
                    kwargs[k] = v
        try:
            result = solve(topic_id, **kwargs)
        except TypeError as exc:
            # A misspelled parameter reaches the solver as an unexpected kwarg.
            # That used to escape this handler and kill the connection with a
            # stack trace and no response at all — `?topic=particle-in-a-box&
            # L_nm=1` was enough to do it. It is a bad request, so say so, and
            # name the parameters the solver actually takes.
            import inspect
            solver = SOLVERS.get(topic_id)
            params = sorted(inspect.signature(solver).parameters) if solver else []
            self._send_json({"error": str(exc), "topic": topic_id,
                             "accepted_parameters": params}, status=400)
            return
        except (ValueError, KeyError) as exc:
            self._send_json({"error": str(exc), "available_solvers": sorted(SOLVERS)}, status=400)
            return
        self._send_json(result)

    def _handle_path(self, query: str) -> None:
        params = parse_qs(query)
        topic_id = params.get("id", [""])[0].strip()
        if not topic_id or topic_id not in TOPICS:
            self._send_json({"error": f"Unknown topic: {topic_id!r}"}, status=404)
            return
        path_ids = learning_path(topic_id)
        self._send_json({
            "target": topic_id,
            "path": [topic_to_dict(TOPICS[t]) for t in path_ids],
        })

    def _handle_problems(self, query: str) -> None:
        params = parse_qs(query)
        topic_id = params.get("topic", [""])[0].strip()
        problems = problems_for_topic(topic_id) if topic_id else []
        self._send_json({"problems": [problem_to_dict(p) for p in problems]})

    def _handle_ask(self, query: str) -> None:
        """Grounded answer, streamed as Server-Sent Events.

        Streaming is not a performance trick here — each stage (match,
        retrieve, compute, compose) is shown to the user as it happens, so the
        provenance of the final answer is watchable rather than asserted.
        """
        params = parse_qs(query)
        question = params.get("q", [""])[0].strip()
        mode = params.get("mode", ["explain"])[0]
        depth = params.get("depth", ["intermediate"])[0]
        if not question:
            self._send_json({"error": "q parameter required"}, status=400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._send_cors_headers()
        self.end_headers()

        def emit(event: str, data: dict) -> None:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
                             .encode("utf-8"))
            self.wfile.flush()

        try:
            # Nine-stage flow: understand → route → gather → evidence →
            # reasoning → professor → validation. It builds on tutor.py, which
            # still owns the solvers, curriculum matching and mastery state.
            try:
                from .qp_pipeline import run as answer_stream
            except ImportError:                        # flat-script execution
                from qp_pipeline import run as answer_stream
            for stage, payload in answer_stream(question, mode=mode, depth=depth):
                emit(stage, payload)
        except (BrokenPipeError, ConnectionResetError):
            return                      # client navigated away mid-answer
        except Exception as exc:        # never leave the stream hanging open
            try:
                emit("error", {"message": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    # ── Math Lab: free-form mathematical input ───────────────────────────
    # Everything below is computed by sympy or numpy. The model never touches
    # these numbers; it may only talk about them.

    def _mathlab(self):
        try:
            from . import mathlab
        except ImportError:
            import mathlab
        return mathlab

    def _handle_math(self, query: str) -> None:
        p = parse_qs(query)
        expr = p.get("expr", [""])[0].strip()
        if not expr:
            self._send_json({"error": "expr parameter required",
                             "operations": self._mathlab().OPERATIONS}, status=400)
            return
        subs = {}
        for pair in p.get("subs", [""])[0].split(","):
            if "=" in pair:
                k, _, v = pair.partition("=")
                subs[k.strip()] = v.strip()
        self._send_json(self._mathlab().evaluate(
            expr,
            operation=p.get("op", ["simplify"])[0],
            variable=p.get("var", ["x"])[0],
            at=(p.get("at", [None])[0]),
            order=int(p.get("order", ["6"])[0] or 6),
            subs=subs or None))

    def _handle_ode(self, query: str) -> None:
        p = parse_qs(query)
        eq = p.get("eq", [""])[0].strip()
        if not eq:
            self._send_json({"error": "eq parameter required"}, status=400)
            return
        self._send_json(self._mathlab().solve_ode(
            eq, func=p.get("func", ["psi"])[0], variable=p.get("var", ["x"])[0]))

    def _handle_schrodinger(self, query: str) -> None:
        p = parse_qs(query)
        params = {}
        for pair in p.get("params", [""])[0].split(","):
            if "=" in pair:
                k, _, v = pair.partition("=")
                try:
                    params[k.strip()] = float(v)
                except ValueError:
                    pass
        try:
            self._send_json(self._mathlab().schrodinger(
                potential=p.get("V", ["0.5*k*x**2"])[0],
                x_min_nm=float(p.get("xmin", ["-5"])[0]),
                x_max_nm=float(p.get("xmax", ["5"])[0]),
                n_points=int(p.get("points", ["1200"])[0]),
                n_states=int(p.get("states", ["5"])[0]),
                mass_me=float(p.get("mass", ["1"])[0]),
                units=p.get("units", ["eV_nm"])[0],
                params=params or None))
        except ValueError as exc:
            self._send_json({"ok": False, "error": f"bad parameter: {exc}"}, status=400)

    def _handle_mathlab_info(self, query: str) -> None:
        """What the Math Lab can do — the UI builds its controls from this."""
        ml = self._mathlab()
        self._send_json({"operations": ml.OPERATIONS,
                         "presets": ml.PRESET_POTENTIALS,
                         "units": ml.UNIT_SYSTEMS})

    def _serve_static(self, path: str) -> None:
        if path == "/":
            target = WEB_ROOT / "standalone.html"
        else:
            target = (WEB_ROOT / path.lstrip("/")).resolve()

        web_root_resolved = WEB_ROOT.resolve()
        if not str(target).startswith(str(web_root_resolved)) or not target.exists():
            self.send_error(404)
            return

        body = target.read_bytes()
        content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the Quantum Professor web UI")
    # A platform (Railway, Heroku) supplies $PORT and requires 0.0.0.0; local
    # runs keep the safer loopback default.
    deployed = "PORT" in os.environ
    parser.add_argument("--host", default="0.0.0.0" if deployed else "127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5052)))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), QuantumHandler)
    print(f"Quantum Professor UI at http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
