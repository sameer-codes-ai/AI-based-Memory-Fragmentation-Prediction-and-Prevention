"""
PREFRAG: Live Interactive Demonstration Launcher
Launches the real-time memory fragmentation & defragmentation demo server
and opens the dashboard in your default browser.
"""

import sys
import os
import argparse
import webbrowser
import threading
import time

# Ensure project root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.demo.live_server import run_server


def open_browser(url: str, delay: float = 0.6):
    def _open():
        time.sleep(delay)
        print(f" [BROWSER] Opening dashboard at: {url}")
        sys.stdout.flush()
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f" [BROWSER] Note: Could not auto-launch browser ({e}).")
            print(f"           Please open {url} manually.")
            sys.stdout.flush()
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    parser = argparse.ArgumentParser(description="PREFRAG Live Memory Fragmentation & Reorganization Demo")
    parser.add_argument("--port", type=int, default=8050, help="HTTP server port (default: 8050)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        open_browser(url)

    run_server(port=args.port)


if __name__ == "__main__":
    main()
