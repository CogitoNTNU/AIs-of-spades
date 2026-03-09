# utils.py
import numpy as np
from collections import Counter
from treys import Card

singulars = [
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Jack",
    "Queen",
    "King",
    "Ace",
]
plurals = [
    "Twos",
    "Threes",
    "Fours",
    "Fives",
    "Sixes",
    "Sevens",
    "Eights",
    "Nines",
    "Tens",
    "Jacks",
    "Queens",
    "Kings",
    "Aces",
]

SUIT_INTS = [1, 2, 4, 8]


def pretty_print_hand(hand_cards, hand_type, table_cards, multiple=False):
    """
    Returns a human-readable description of the best hand.

    hand_cards:  list of 2 hole card ints
    hand_type:   string from treys evaluator (e.g. 'Full House')
    table_cards: list of 3-5 community card ints
    multiple:    unused legacy param, kept for backwards compatibility
    """
    combined = list(hand_cards) + list(table_cards)
    values = [Card.get_rank_int(c) for c in combined]
    suits = [Card.get_suit_int(c) for c in combined]

    if hand_type == "High Card":
        return "high card %s" % singulars[max(values)]

    if hand_type == "Pair":
        pair_ranks = [k for k, v in Counter(values).items() if v >= 2]
        return "a pair of %s" % plurals[max(pair_ranks)]

    if hand_type == "Two Pair":
        pair_ranks = [k for k, v in Counter(values).items() if v >= 2]
        pair_ranks.sort(reverse=True)
        return "two pair, %s and %s" % (plurals[pair_ranks[0]], plurals[pair_ranks[1]])

    if hand_type == "Three of a Kind":
        triple_ranks = [k for k, v in Counter(values).items() if v >= 3]
        return "three of a kind, %s" % plurals[max(triple_ranks)]

    if hand_type == "Straight":
        low, high = _find_straight(values)
        if low is None:
            raise Exception("Could not find straight in values: %s" % values)
        return "a straight, %s to %s" % (singulars[low], singulars[high])

    if hand_type == "Flush":
        suit_i = _dominant_suit_index(suits)
        flush_values = [
            values[i] for i in range(len(values)) if suits[i] == SUIT_INTS[suit_i]
        ]
        return "a flush, %s high" % singulars[max(flush_values)]

    if hand_type == "Full House":
        triple_ranks = [k for k, v in Counter(values).items() if v >= 3]
        # Exclude the triple rank from doubles to avoid "Kings full of Kings"
        pair_ranks = [
            k for k, v in Counter(values).items() if v >= 2 and k not in triple_ranks
        ]
        return "a full house, %s full of %s" % (
            plurals[max(triple_ranks)],
            plurals[max(pair_ranks)],
        )

    if hand_type == "Four of a Kind":
        quad_ranks = [k for k, v in Counter(values).items() if v >= 4]
        return "four of a kind, %s" % plurals[max(quad_ranks)]

    if hand_type == "Straight Flush":
        suit_i = _dominant_suit_index(suits)
        suited_values = [
            values[i] for i in range(len(values)) if suits[i] == SUIT_INTS[suit_i]
        ]
        low, high = _find_straight(suited_values)
        if low is None:
            raise Exception(
                "Could not find straight flush in suited values: %s" % suited_values
            )
        # Royal flush: Ace-high straight flush
        if high == 12:
            return "a royal flush"
        return "a straight flush, %s to %s" % (singulars[low], singulars[high])

    raise Exception(
        "Unrecognized hand type '%s' passed to pretty_print_hand" % hand_type
    )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _find_straight(values: list) -> tuple:
    """
    Finds the highest 5-card straight within values.
    Returns (low_rank, high_rank) or (None, None) if no straight found.
    Handles the wheel (A-2-3-4-5): Ace counts as rank -1 below Two.
    """
    unique = sorted(set(values), reverse=True)

    # Broadway check first (high to low) for normal straights
    for i in range(len(unique) - 4):
        window = unique[i : i + 5]
        if window[0] - window[4] == 4 and len(set(window)) == 5:
            return window[4], window[0]

    # Wheel: A-2-3-4-5 (Ace acts as -1, i.e. below Two which is rank 0)
    if 12 in unique:  # Ace present
        low_four = [v for v in unique if v <= 3]  # 2,3,4,5 are ranks 0,1,2,3
        if len(low_four) >= 4 and sorted(low_four)[:4] == [0, 1, 2, 3]:
            return 0, 3  # Five-high straight: Two to Five (Ace plays low)

    return None, None


def _dominant_suit_index(suits: list) -> int:
    """Returns the index into SUIT_INTS of the most frequent suit."""
    counts = [suits.count(s) for s in SUIT_INTS]
    return int(np.argmax(counts))


# ------------------------------------------------------------------
# Approximate comparisons (used by BettingManager)
# ------------------------------------------------------------------


def approx_lte(x, y) -> bool:
    return x <= y or np.isclose(x, y)


def approx_gt(x, y) -> bool:
    return x > y and not np.isclose(x, y)
