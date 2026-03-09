"""
poker_ui.py — Interactive poker table debugger (Tkinter, no extra dependencies)

Usage:
    python poker_ui.py
    python poker_ui.py --players 4
"""

import tkinter as tk
from tkinter import ttk, font
import argparse
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_table import TestTable
from pokerenv.common import PlayerAction


SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_COLORS = {"s": "#d0d0d0", "h": "#e05555", "d": "#e05555", "c": "#d0d0d0"}
BG = "#0a0f1a"
BG2 = "#111823"
GOLD = "#d4af37"
BLUE = "#4a9eff"
MUTED = "#555e6e"
TEXT = "#e8e8e8"
GREEN_FELT = "#0d3320"


def card_color(card_str):
    if len(card_str) < 2:
        return TEXT
    return SUIT_COLORS.get(card_str[-1], TEXT)


def fmt_card(card_str):
    if len(card_str) < 2:
        return card_str
    rank, suit = card_str[:-1], card_str[-1]
    return rank + SUIT_SYMBOLS.get(suit, suit)


def fmt_bb(val_in_bb):
    return f"${val_in_bb * 5:.2f}"


class PokerUI:
    def __init__(self, root, n_players=3):
        self.root = root
        self.root.title("Poker Table — Debug UI")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.n_players = n_players
        self.table = TestTable(n_players=n_players)
        self.snapshot = None
        self.viewing_as = 0

        self._build_ui()
        self._new_hand()

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_header()

        body = tk.Frame(self.root, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        # Left: table + controls
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=10)
        left.columnconfigure(0, weight=1)

        self._build_street_bar(left)
        self._build_community(left)
        self._build_players(left)
        self._build_actions(left)

        # Right: log
        right = tk.Frame(body, bg=BG2, width=260)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=0)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        body.columnconfigure(1, minsize=260)
        self._build_log(right)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg="#060c16", pady=8)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        tk.Label(
            hdr,
            text="POKER",
            bg="#060c16",
            fg=GOLD,
            font=("Courier", 16, "bold"),
            padx=16,
        ).grid(row=0, column=0)
        tk.Label(
            hdr, text="TABLE DEBUG", bg="#060c16", fg=MUTED, font=("Courier", 9), padx=0
        ).grid(row=0, column=1, sticky="w")

        ctrl = tk.Frame(hdr, bg="#060c16")
        ctrl.grid(row=0, column=2, padx=12)

        tk.Label(
            ctrl, text="Players:", bg="#060c16", fg=MUTED, font=("Courier", 10)
        ).pack(side="left", padx=(0, 4))
        self.players_var = tk.IntVar(value=self.n_players)
        for n in range(2, 7):
            rb = tk.Radiobutton(
                ctrl,
                text=str(n),
                variable=self.players_var,
                value=n,
                bg="#060c16",
                fg=TEXT,
                selectcolor=BG,
                activebackground="#060c16",
                activeforeground=GOLD,
                font=("Courier", 10),
                command=self._on_nplayers_change,
            )
            rb.pack(side="left", padx=2)

        tk.Button(
            ctrl,
            text="New Hand",
            bg="#1a1400",
            fg=GOLD,
            font=("Courier", 10),
            relief="flat",
            bd=0,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._new_hand,
        ).pack(side="left", padx=(12, 0))

        tk.Button(
            ctrl,
            text="Show All",
            bg="#101520",
            fg=BLUE,
            font=("Courier", 10),
            relief="flat",
            bd=0,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._toggle_show_all,
        ).pack(side="left", padx=4)
        self.show_all = False

    def _build_street_bar(self, parent):
        self.street_frame = tk.Frame(parent, bg=BG2, pady=6)
        self.street_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.street_frame.columnconfigure(5, weight=1)
        self.street_labels = {}
        streets = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
        for i, s in enumerate(streets):
            lbl = tk.Label(
                self.street_frame,
                text=s,
                bg=BG2,
                fg=MUTED,
                font=("Courier", 9),
                padx=8,
                pady=2,
            )
            lbl.grid(row=0, column=i, padx=2)
            self.street_labels[s] = lbl

        self.pot_label = tk.Label(
            self.street_frame,
            text="POT —",
            bg=BG2,
            fg=GOLD,
            font=("Courier", 11, "bold"),
        )
        self.pot_label.grid(row=0, column=6, padx=(20, 4))
        self.match_label = tk.Label(
            self.street_frame, text="TO MATCH —", bg=BG2, fg=TEXT, font=("Courier", 10)
        )
        self.match_label.grid(row=0, column=7, padx=4)
        self.raise_label = tk.Label(
            self.street_frame,
            text="MIN RAISE —",
            bg=BG2,
            fg=MUTED,
            font=("Courier", 10),
        )
        self.raise_label.grid(row=0, column=8, padx=(4, 12))

    def _build_community(self, parent):
        felt = tk.Frame(parent, bg=GREEN_FELT, pady=14)
        felt.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        felt.columnconfigure(0, weight=1)

        inner = tk.Frame(felt, bg=GREEN_FELT)
        inner.grid(row=0, column=0)
        self.community_frames = []
        for i in range(5):
            f = tk.Frame(inner, bg="#0a2818", width=52, height=72, relief="flat", bd=1)
            f.grid(row=0, column=i, padx=5)
            f.grid_propagate(False)
            lbl = tk.Label(
                f,
                text="",
                bg="#0a2818",
                fg=TEXT,
                font=("Courier", 18, "bold"),
                width=3,
                height=2,
            )
            lbl.place(relx=0.5, rely=0.5, anchor="center")
            self.community_frames.append((f, lbl))

    def _build_players(self, parent):
        self.players_frame = tk.Frame(parent, bg=BG)
        self.players_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.player_widgets = []  # list of dicts per player

    def _rebuild_player_seats(self):
        for w in self.players_frame.winfo_children():
            w.destroy()
        self.player_widgets = []
        n = len(self.snapshot["players"])
        for i in range(n):
            self.players_frame.columnconfigure(i, weight=1)

        for i, p in enumerate(self.snapshot["players"]):
            frame = tk.Frame(
                self.players_frame, bg=BG2, pady=8, padx=10, relief="flat", bd=1
            )
            frame.grid(row=0, column=i, padx=5, sticky="nsew")
            frame.bind("<Button-1>", lambda e, idx=i: self._set_view(idx))

            name_lbl = tk.Label(
                frame, text=p["name"], bg=BG2, fg=TEXT, font=("Courier", 11, "bold")
            )
            name_lbl.grid(row=0, column=0, columnspan=2, sticky="w")
            name_lbl.bind("<Button-1>", lambda e, idx=i: self._set_view(idx))

            you_lbl = tk.Label(frame, text="", bg=BG2, fg=BLUE, font=("Courier", 8))
            you_lbl.grid(row=0, column=2, sticky="e")

            cards_frame = tk.Frame(frame, bg=BG2)
            cards_frame.grid(row=1, column=0, columnspan=3, pady=4)
            c1 = tk.Label(
                cards_frame,
                text="",
                bg="#1a2a3a",
                fg=TEXT,
                font=("Courier", 14, "bold"),
                width=3,
                height=1,
                relief="flat",
                padx=4,
                pady=2,
            )
            c1.grid(row=0, column=0, padx=2)
            c2 = tk.Label(
                cards_frame,
                text="",
                bg="#1a2a3a",
                fg=TEXT,
                font=("Courier", 14, "bold"),
                width=3,
                height=1,
                relief="flat",
                padx=4,
                pady=2,
            )
            c2.grid(row=0, column=1, padx=2)

            stack_lbl = tk.Label(frame, text="", bg=BG2, fg=TEXT, font=("Courier", 9))
            pot_lbl = tk.Label(frame, text="", bg=BG2, fg=MUTED, font=("Courier", 9))
            state_lbl = tk.Label(frame, text="", bg=BG2, fg=MUTED, font=("Courier", 8))
            win_lbl = tk.Label(
                frame, text="", bg=BG2, fg="#7ec87e", font=("Courier", 9, "bold")
            )

            stack_lbl.grid(row=2, column=0, columnspan=3, sticky="w")
            pot_lbl.grid(row=3, column=0, columnspan=3, sticky="w")
            state_lbl.grid(row=4, column=0, columnspan=3, sticky="w")
            win_lbl.grid(row=5, column=0, columnspan=3, sticky="w")

            self.player_widgets.append(
                {
                    "frame": frame,
                    "name": name_lbl,
                    "you": you_lbl,
                    "c1": c1,
                    "c2": c2,
                    "stack": stack_lbl,
                    "pot": pot_lbl,
                    "state": state_lbl,
                    "win": win_lbl,
                }
            )

    def _build_actions(self, parent):
        self.action_frame = tk.Frame(parent, bg=BG2, pady=10, padx=14)
        self.action_frame.grid(row=3, column=0, sticky="ew")

        self.acting_label = tk.Label(
            self.action_frame, text="", bg=BG2, fg=GOLD, font=("Courier", 11)
        )
        self.acting_label.grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 6))

        self.fold_btn = tk.Button(
            self.action_frame,
            text="Fold",
            width=8,
            bg="#2a0808",
            fg="#e07070",
            font=("Courier", 11),
            relief="flat",
            cursor="hand2",
            command=self._do_fold,
        )
        self.fold_btn.grid(row=1, column=0, padx=(0, 6))

        self.call_btn = tk.Button(
            self.action_frame,
            text="Call",
            width=10,
            bg="#08162a",
            fg="#70aae0",
            font=("Courier", 11),
            relief="flat",
            cursor="hand2",
            command=self._do_call,
        )
        self.call_btn.grid(row=1, column=1, padx=6)

        self.bet_entry = tk.Entry(
            self.action_frame,
            width=10,
            bg="#1a1a2a",
            fg=TEXT,
            font=("Courier", 11),
            insertbackground=TEXT,
            relief="flat",
        )
        self.bet_entry.grid(row=1, column=2, padx=6)

        self.bet_btn = tk.Button(
            self.action_frame,
            text="Bet/Raise",
            width=10,
            bg="#1a1400",
            fg=GOLD,
            font=("Courier", 11),
            relief="flat",
            cursor="hand2",
            command=self._do_bet,
        )
        self.bet_btn.grid(row=1, column=3, padx=6)

        self.allin_btn = tk.Button(
            self.action_frame,
            text="All-in",
            width=7,
            bg="#120a00",
            fg="#a08030",
            font=("Courier", 10),
            relief="flat",
            cursor="hand2",
            command=self._do_allin,
        )
        self.allin_btn.grid(row=1, column=4, padx=6)

        self.range_label = tk.Label(
            self.action_frame, text="", bg=BG2, fg=MUTED, font=("Courier", 9)
        )
        self.range_label.grid(row=2, column=0, columnspan=5, sticky="w", pady=(4, 0))

    def _build_log(self, parent):
        tk.Label(
            parent, text="HAND LOG", bg=BG2, fg=MUTED, font=("Courier", 9), pady=8
        ).grid(row=0, column=0, sticky="w", padx=12)

        self.log_text = tk.Text(
            parent,
            bg=BG2,
            fg=MUTED,
            font=("Courier", 9),
            relief="flat",
            bd=0,
            wrap="word",
            state="disabled",
            width=32,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        parent.rowconfigure(1, weight=1)

        sb = tk.Scrollbar(parent, command=self.log_text.yview, bg=BG2)
        sb.grid(row=1, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=sb.set)

        # Tag colors
        self.log_text.tag_configure("street", foreground=GOLD)
        self.log_text.tag_configure("sep", foreground="#333")
        self.log_text.tag_configure("win", foreground="#7ec87e")
        self.log_text.tag_configure("normal", foreground="#778899")

    # ──────────────────────────────────────────────────────────────────
    # State update
    # ──────────────────────────────────────────────────────────────────

    def _refresh(self):
        s = self.snapshot

        # Street bar
        streets = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
        for name, lbl in self.street_labels.items():
            active = name == s["street"]
            lbl.configure(
                fg=GOLD if active else MUTED,
                font=("Courier", 9, "bold") if active else ("Courier", 9),
            )
        self.pot_label.configure(text="POT %s" % fmt_bb(s["pot"]))
        self.match_label.configure(text="TO MATCH %s" % fmt_bb(s["bet_to_match"]))
        self.raise_label.configure(text="MIN RAISE %s" % fmt_bb(s["minimum_raise"]))

        # Community cards
        cc = s["community_cards"]
        shown = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5, "SHOWDOWN": 5}.get(
            s["street"], 0
        )
        for i, (frame, lbl) in enumerate(self.community_frames):
            if i < shown and i < len(cc):
                cs = cc[i]
                lbl.configure(text=fmt_card(cs), fg=card_color(cs), bg="#f0ede6")
                frame.configure(bg="#f0ede6")
            else:
                lbl.configure(text="", bg="#0a2818")
                frame.configure(bg="#0a2818")

        # Players — rebuild seats if player count changed
        if len(self.player_widgets) != len(s["players"]):
            self._rebuild_player_seats()

        for i, (p, w) in enumerate(zip(s["players"], self.player_widgets)):
            is_acting = (p["identifier"] == s["acting_player"]) and not s["done"]
            is_viewing = p["identifier"] == self.viewing_as
            folded = p["state"] == "FOLDED"

            bg = "#1a2a1a" if is_acting else ("#101828" if is_viewing else BG2)
            border_color = GOLD if is_acting else (BLUE if is_viewing else BG2)
            w["frame"].configure(
                bg=bg,
                highlightbackground=border_color,
                highlightthickness=1 if (is_acting or is_viewing) else 0,
            )
            for child in w["frame"].winfo_children():
                try:
                    child.configure(bg=bg)
                except:
                    pass

            w["name"].configure(
                fg=GOLD if is_acting else TEXT, font=("Courier", 11, "bold")
            )
            w["you"].configure(text="YOU" if is_viewing else "")

            show = self.show_all or is_viewing or s["done"]
            cards = p["cards"]
            if show and len(cards) == 2:
                for j, (key, cs) in enumerate(zip(["c1", "c2"], cards)):
                    w[key].configure(
                        text=fmt_card(cs),
                        fg=card_color(cs),
                        bg="#f0ede6" if not folded else "#2a2a2a",
                    )
            else:
                for key in ["c1", "c2"]:
                    w[key].configure(text="★", fg="#1a3a5c", bg="#0d2030")

            w["stack"].configure(text="Stack  %s" % fmt_bb(p["stack"]))
            w["pot"].configure(text="In pot %s" % fmt_bb(p["money_in_pot"]))
            w["state"].configure(
                text=p["state"] + (" ALL-IN" if p["all_in"] else ""),
                fg="#e05555" if folded else (GOLD if p["all_in"] else MUTED),
            )
            if s["done"] and p["winnings"] != 0:
                sign = "+" if p["winnings"] > 0 else ""
                w["win"].configure(text="%s%s" % (sign, fmt_bb(p["winnings"])))
            else:
                w["win"].configure(text="")

        # Action controls
        if s["done"]:
            self.acting_label.configure(text="★  Hand complete — press New Hand")
            self._set_buttons_state("disabled")
        else:
            acting_name = (
                s["players"][s["acting_player"]]["name"]
                if s["acting_player"] is not None
                else "?"
            )
            self.acting_label.configure(text="%s to act" % acting_name)
            va = s["valid_actions"]
            actions = va["actions"] if va else []
            lo, hi = va["bet_range"] if va else [0, 0]

            self.fold_btn.configure(state="normal" if "FOLD" in actions else "disabled")
            self.call_btn.configure(
                state="normal" if "CALL" in actions else "disabled",
                text=(
                    "Check"
                    if s["bet_to_match"] == 0
                    else "Call %s"
                    % fmt_bb(
                        min(
                            s["bet_to_match"]
                            - s["players"][s["acting_player"]]["bet_this_street"],
                            s["players"][s["acting_player"]]["stack"],
                        )
                    )
                ),
            )
            self.bet_btn.configure(
                state="normal" if "BET" in actions else "disabled",
                text="Raise" if s["bet_to_match"] > 0 else "Bet",
            )
            self.allin_btn.configure(state="normal" if "BET" in actions else "disabled")
            self.range_label.configure(
                text=(
                    "Bet range: %s – %s" % (fmt_bb(lo), fmt_bb(hi))
                    if "BET" in actions
                    else ""
                )
            )
            if "BET" in actions and not self.bet_entry.get():
                self.bet_entry.delete(0, "end")
                self.bet_entry.insert(0, "%.1f" % lo)

        # Log
        self._refresh_log(s["hand_log"])

    def _refresh_log(self, lines):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for line in lines:
            if line.startswith("***"):
                self.log_text.insert("end", line + "\n", "street")
            elif line.startswith("---"):
                self.log_text.insert("end", line + "\n", "sep")
            elif "wins" in line or "collected" in line:
                self.log_text.insert("end", line + "\n", "win")
            else:
                self.log_text.insert("end", line + "\n", "normal")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_buttons_state(self, state):
        for btn in [self.fold_btn, self.call_btn, self.bet_btn, self.allin_btn]:
            btn.configure(state=state)

    # ──────────────────────────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────────────────────────

    def _new_hand(self):
        self.n_players = self.players_var.get()
        self.table = TestTable(n_players=self.n_players)
        self.snapshot = self.table.reset()
        self.viewing_as = self.snapshot["acting_player"] or 0
        self._refresh()

    def _on_nplayers_change(self):
        self._new_hand()

    def _toggle_show_all(self):
        self.show_all = not self.show_all
        self._refresh()

    def _set_view(self, idx):
        self.viewing_as = idx
        self._refresh()

    def _do_fold(self):
        self.snapshot = self.table.fold()
        self._refresh()

    def _do_call(self):
        self.snapshot = self.table.call()
        self._refresh()

    def _do_bet(self):
        try:
            amount = float(self.bet_entry.get())
        except ValueError:
            self.range_label.configure(text="⚠  Enter a valid number", fg="#e07070")
            return
        self.snapshot = self.table.bet(amount)
        self.bet_entry.delete(0, "end")
        self._refresh()

    def _do_allin(self):
        acting_i = self.snapshot["acting_player"]
        if acting_i is None:
            return
        stack = self.snapshot["players"][acting_i]["stack"]
        self.snapshot = self.table.bet(stack)
        self._refresh()


# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Poker Table Debug UI")
    parser.add_argument(
        "--players",
        type=int,
        default=3,
        choices=range(2, 7),
        help="Number of players (2–6)",
    )
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("1100x680")
    app = PokerUI(root, n_players=args.players)
    root.mainloop()


if __name__ == "__main__":
    main()
