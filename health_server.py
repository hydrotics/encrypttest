import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import urlsplit


class _HealthRequestHandler(BaseHTTPRequestHandler):
    server: "_HealthHTTPServer"
    server_version = "RevealBotHealth/1.1"

    def do_GET(self) -> None:
        self._handle_request(send_body=True)

    def do_HEAD(self) -> None:
        # UptimeRobot HTTP monitors use HEAD by default.
        # BaseHTTPRequestHandler returns 501 unless do_HEAD() is implemented.
        self._handle_request(send_body=False)

    def _handle_request(self, send_body: bool) -> None:
        path = urlsplit(self.path).path

        # These endpoints always report that the HTTP server itself is alive.
        # Discord readiness is included in the response but does not change
        # the health endpoint's 200 status.
        if path in {"/", "/health", "/healthz"}:
            ready = self._readiness()
            self._respond(
                200,
                {
                    "status": "ok",
                    "discord_ready": ready,
                },
                send_body=send_body,
            )
            return

        # This endpoint reflects actual Discord readiness.
        if path == "/readyz":
            ready = self._readiness()
            status = 200 if ready else 503

            self._respond(
                status,
                {
                    "status": "ready" if ready else "not_ready",
                },
                send_body=send_body,
            )
            return

        self._respond(
            404,
            {"status": "not_found"},
            send_body=send_body,
        )

    def _readiness(self) -> bool:
        try:
            return bool(self.server.readiness_check())
        except Exception:
            # Don't let a broken Discord readiness check kill the HTTP
            # health server. /readyz will become 503, while / and /health
            # can still be used to confirm the process is alive.
            return False

    def _respond(
        self,
        status: int,
        payload: dict,
        send_body: bool = True,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        # HEAD must return headers but no response body.
        if send_body:
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

    def log_message(self, format: str, *args: object) -> None:
        # Silence default HTTP request logging.
        return


class _HealthHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        readiness_check: Callable[[], bool],
    ) -> None:
        self.readiness_check = readiness_check
        super().__init__(address, _HealthRequestHandler)


def start_health_server(
    readiness_check: Callable[[], bool],
    host: str | None = None,
    port: int | None = None,
) -> _HealthHTTPServer:
    # Render requires the server to bind to 0.0.0.0 and recommends using
    # the PORT environment variable. Render's default PORT is 10000.
    bind_host = host or os.getenv("HEALTH_HOST", "0.0.0.0")
    bind_port = (
        int(os.getenv("PORT", "10000"))
        if port is None
        else port
    )

    server = _HealthHTTPServer(
        (bind_host, bind_port),
        readiness_check,
    )

    threading.Thread(
        target=server.serve_forever,
        name="health-http",
        daemon=True,
    ).start()

    return server
