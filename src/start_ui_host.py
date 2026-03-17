import argparse

from ui.ui_table_manager import UITableManager


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poker LAN server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--http", type=int, default=8080)
    parser.add_argument("--seats", type=int, default=4)
    parser.add_argument("--hands", type=int, default=20)
    parser.add_argument("--stack-low", type=int, default=50)
    parser.add_argument("--stack-high", type=int, default=200)
    args = parser.parse_args()

    UITableManager(
        n_seats=args.seats,
        host=args.host,
        ws_port=args.port,
        http_port=args.http,
        total_hands=args.hands,
        stack_low=args.stack_low,
        stack_high=args.stack_high,
    ).run()
