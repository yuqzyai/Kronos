#!/usr/bin/env python3
"""Start the local Kronos web UI.

Does not install packages, open a browser, enable the debugger, or bind a
public interface. Install dependencies yourself inside the existing virtualenv.
"""

import os
import sys


def main():
    os.environ.pop("FLASK_DEBUG", None)
    try:
        from app import app, server_options
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("From the repository root, with the project virtualenv active:")
        print("  pip install -r requirements.txt -r webui/requirements.txt")
        return 1
    options = server_options()
    print(f"Kronos Web UI at http://{options['host']}:{options['port']}")
    print("Localhost only. Debugger and reloader are off.")
    app.run(**options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
