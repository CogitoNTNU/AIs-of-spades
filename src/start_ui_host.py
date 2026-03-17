import argparse

from ui.ui_table_manager import UITableManager


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
        help="Percorso ai pesi di UGO (es. ui_weights/even_net/weights.pt). "
        "Se omesso, tutti i seat sono umani.",
    )
    args = parser.parse_args()

    manager = UITableManager(
        n_seats=args.seats,
        host=args.host,
        ws_port=args.port,
        http_port=args.http,
        total_hands=args.hands,
        stack_low=args.stack_low,
        stack_high=args.stack_high,
    )

    if args.ugo:
        from nn.even_net import EvenNet
        from ui.ui_agent_player import AIPlayer

        UGO_SEAT = 0
        manager.players[UGO_SEAT] = AIPlayer(
            seat=UGO_SEAT,
            name="UGO",
            model_class=EvenNet,
            weights_path=args.ugo,
        )
        manager._name_to_seat["ugo"] = UGO_SEAT

    manager.run()
