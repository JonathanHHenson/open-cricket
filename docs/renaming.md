# Moving from LabelJudge to open-cricket

[Documentation index](README.md)

The project is now Open Cricket. This is a breaking package/interface rename;
legacy imports, commands, and environment variables are not retained as aliases.

| Previous name | New name |
| --- | --- |
| Distribution `labeljudge` | `open-cricket` |
| Python package `labeljudge` | `open_cricket` |
| CLI `labeljudge` | `open-cricket` |
| `LABELJUDGE_MODEL` | `OPEN_CRICKET_MODEL` |
| `LABELJUDGE_BACKEND` | `OPEN_CRICKET_BACKEND` |
| `LABELJUDGE_DEVICE` | `OPEN_CRICKET_DEVICE` |
| `LABELJUDGE_REVISION` | `OPEN_CRICKET_REVISION` |
| `LABELJUDGE_API_KEY` | `OPEN_CRICKET_API_KEY` |
| `x-labeljudge-confidence-method` | `x-open-cricket-confidence-method` |
| Injected default model `labeljudge-custom` | `open-cricket-custom` |

Update the configured API key variable when upgrading an authenticated server;
the old variable no longer enables authentication. Start the server with
`uvicorn open_cricket.server:app`. Python applications should use imports such as
`from open_cricket import Client, LocalClient, Choice, Score, Noul`.

The HTTP routes, request/response bodies, default checkpoint, and scoring behavior
are unchanged. Reinstall the checkout with the extras you need to refresh the
CLI entry point, for example `uv sync --extra hf --extra server`.
