import argparse
import logging
from pathlib import Path

from nn import MODEL_CLASSES

from ui.ui_table_manager import UITableManager
from ui.ui_agent_player import AIPlayer

log = logging.getLogger("start_ui_host")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poker LAN server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--http", type=int, default=8080)
    parser.add_argument("--seats", type=int, default=4)
    parser.add_argument("--hands", type=int, default=20)
    parser.add_argument("--stack-low", type=int, default=50)
    parser.add_argument("--stack-high", type=int, default=200)
    parser.add_argument(
        "--ugo",
        metavar="WEIGHTS",
        default=None,
        help="UGO's weights path (es. ui_weights/even_net/weights.pt). "
        "If none all the seats are human",
    )
    parser.add_argument(
        "--ugo-model",
        default="EvenNet",
        type=str,
        help="Change the UGO's type of network [default EvenNet]",
    )
    parser.add_argument(
        "--no-auto-reset",
        action="store_true",
        default=False,
        help="Return to lobby instead of resetting the table when one player remains or hands run out",
    )
    parser.add_argument(
        "--opponents-dir",
        default=None,
        metavar="DIR",
        help="Directory containing AI opponents for AI-vs-AI mode: DIR/[model_name]/[weights].pt "
             "(e.g. ui_weights/opponents). If not set, the server runs in human lobby mode.",
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds to wait between actions (default: 5 when using --opponents-dir, 0 otherwise)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Scan opponents directory
    # ------------------------------------------------------------------
    opponents = []
    opponents_dir = Path(args.opponents_dir) if args.opponents_dir else None
    if opponents_dir is not None and opponents_dir.is_dir():
        for model_dir in sorted(opponents_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_key = "".join(w.capitalize() for w in model_dir.name.split("_"))
            if model_key not in MODEL_CLASSES:
                log.warning("Unknown model folder '%s' (no class '%s' in MODEL_CLASSES — skipped)", model_dir.name, model_key)
                continue
            model_class = MODEL_CLASSES[model_key]
            for weights_file in sorted(model_dir.glob("*.pt")):
                name = f"{model_key}-{weights_file.stem}"
                opponents.append((model_class, weights_file, name))

    # ------------------------------------------------------------------
    # Build manager
    # ------------------------------------------------------------------
    if opponents:
        n_seats = len(opponents)
        action_delay = args.action_delay if args.action_delay is not None else 5.0
        log.info(
            "Found %d opponent(s) in '%s' — starting AI-vs-AI with %.0fs action delay",
            n_seats, args.opponents_dir, action_delay,
        )
        manager = UITableManager(
            n_seats=n_seats,
            host=args.host,
            ws_port=args.port,
            http_port=args.http,
            total_hands=args.hands,
            stack_low=args.stack_low,
            stack_high=args.stack_high,
            auto_reset=not args.no_auto_reset,
            action_delay=action_delay,
        )
        for seat, (model_class, weights_file, name) in enumerate(opponents):
            log.info("  Seat %d  ->  %s  (%s)", seat, name, weights_file)
            manager.players[seat] = AIPlayer(
                seat=seat,
                name=name,
                model_class=model_class,
                weights_path=str(weights_file),
            )
            manager._name_to_seat[name.lower()] = seat
    else:
        action_delay = args.action_delay if args.action_delay is not None else 0.0
        manager = UITableManager(
            n_seats=args.seats,
            host=args.host,
            ws_port=args.port,
            http_port=args.http,
            total_hands=args.hands,
            stack_low=args.stack_low,
            stack_high=args.stack_high,
            auto_reset=not args.no_auto_reset,
            action_delay=action_delay,
        )
        if args.ugo:
            model_class = MODEL_CLASSES[args.ugo_model]
            UGO_SEAT = 0
            manager.players[UGO_SEAT] = AIPlayer(
                seat=UGO_SEAT,
                name="UGO",
                model_class=model_class,
                weights_path=args.ugo,
            )
            manager._name_to_seat["ugo"] = UGO_SEAT

    manager.run()
