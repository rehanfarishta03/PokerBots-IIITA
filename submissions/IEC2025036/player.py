'''
Poker Bot — Tight-Aggressive (TAG) Strategy v0.2
=================================================
A deterministic, rule-based Texas Hold'em poker bot for the IIITA Bounty
Poker engine. Uses eval7 for post-flop hand evaluation.

v0.2: Added deterministic bluffing via hash-based pseudo-randomness.

Every decision is O(1) — frozenset lookups and integer comparisons only.
No randomness, no simulation, no learning.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

try:
    import eval7
except ImportError:
    import eval7_fallback as eval7  # Pure Python fallback for local dev


# ---------------------------------------------------------------------------
# RANK ORDERING — used for canonical hand notation & top-pair detection
# ---------------------------------------------------------------------------
_RANK_ORDER: dict[str, int] = {
    'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
    '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
    '4': 4, '3': 3, '2': 2,
}

# ---------------------------------------------------------------------------
# TOP ~20% STARTING HANDS — Premium TAG Range (frozenset for O(1) lookup)
# ---------------------------------------------------------------------------
# Notation: "AKs" = suited, "AKo" = offsuit, "AA" = pocket pair
_PREMIUM_HANDS: frozenset[str] = frozenset({
    # --- Pocket Pairs (9) ---
    'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66',
    # --- Suited Aces (10) ---
    'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A5s', 'A4s', 'A3s', 'A2s',
    # --- Suited Broadway + Connectors (17) ---
    'KQs', 'KJs', 'KTs', 'K9s',
    'QJs', 'QTs', 'Q9s',
    'JTs', 'J9s',
    'T9s', 'T8s',
    '98s', '97s',
    '87s', '76s', '65s', '54s',
    # --- Offsuit Broadway (7) ---
    'AKo', 'AQo', 'AJo', 'ATo',
    'KQo', 'KJo',
    'QJo',
})

# ---------------------------------------------------------------------------
# POST-FLOP HAND STRENGTH TIERS
# ---------------------------------------------------------------------------
# eval7.handtype() returns these strings. Two Pair+ is always strong.
# 'Pair' is handled separately (top pair vs. weak pair).
_STRONG_HAND_TYPES: frozenset[str] = frozenset({
    'Two Pair',
    'Trips',
    'Straight',
    'Flush',
    'Full House',
    'Quads',
    'Straight Flush',
})


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS — module-level for O(1) access
# ---------------------------------------------------------------------------

def _canonicalize_hand(card1: str, card2: str) -> str:
    """
    Convert two hole cards into canonical hand notation.
    ('Ah', 'Kh') → 'AKs', ('Kd', 'Ah') → 'AKo', ('Jh', 'Jd') → 'JJ'
    """
    rank1, suit1 = card1[0], card1[1]
    rank2, suit2 = card2[0], card2[1]

    # Ensure higher rank comes first
    if _RANK_ORDER[rank1] < _RANK_ORDER[rank2]:
        rank1, suit1, rank2, suit2 = rank2, suit2, rank1, suit1

    # Pocket pair — no suit suffix
    if rank1 == rank2:
        return rank1 + rank2

    # Suited vs offsuit
    suffix = 's' if suit1 == suit2 else 'o'
    return rank1 + rank2 + suffix


def _is_top_pair_or_overpair(hole_cards: list[str], board_cards: list[str]) -> bool:
    """
    True if we have top pair (paired with highest board card)
    or an overpair (pocket pair above all board cards).
    """
    hole_rank1: int = _RANK_ORDER[hole_cards[0][0]]
    hole_rank2: int = _RANK_ORDER[hole_cards[1][0]]
    board_top: int = max(_RANK_ORDER[c[0]] for c in board_cards)

    # Overpair: pocket pair above all board cards
    if hole_rank1 == hole_rank2 and hole_rank1 > board_top:
        return True

    # Top pair: one hole card matches the highest board card
    if hole_rank1 == board_top or hole_rank2 == board_top:
        return True

    return False


# ---------------------------------------------------------------------------
# PLAYER CLASS — integrates TAG strategy with the skeleton engine
# ---------------------------------------------------------------------------

class Player(Bot):
    '''
    Tight-Aggressive poker bot with deterministic, O(1) decision-making.

    Pre-flop:  Raise with top 20% hands, fold everything else.
    Post-flop: Raise with strong made hands (Two Pair+, Top Pair),
               check/fold with weak ones.
    Bounty:    Widen range slightly when we hold the bounty rank.
    '''

    def __init__(self):
        '''Called when a new game starts. Called exactly once.'''
        self.bounty_rank = None   # Current bounty rank for this set of rounds

    def handle_new_round(self, game_state, round_state, active):
        '''Called when a new round starts. Called NUM_ROUNDS times.'''
        self.bounty_rank = round_state.bounties[active]

    def handle_round_over(self, game_state, terminal_state, active):
        '''Called when a round ends. Called NUM_ROUNDS times.'''
        pass

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens — deterministic TAG decision tree.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        FoldAction(), CallAction(), CheckAction(), or RaiseAction(amount).
        '''
        # --- Extract game state ---
        legal_actions = round_state.legal_actions()
        street = round_state.street           # 0=preflop, 3=flop, 4=turn, 5=river
        my_cards = round_state.hands[active]   # ['Ah', 'Kd']
        board_cards = round_state.deck[:street] # community cards
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        continue_cost = opp_pip - my_pip       # chips needed to stay in
        pot_size = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

        # --- Compute raise bounds if raising is legal ---
        min_raise = 0
        max_raise = 0
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

        # --- Bounty awareness ---
        has_bounty = (
            my_cards[0][0] == self.bounty_rank or
            my_cards[1][0] == self.bounty_rank
        )

        # --- Clock safety: ultra-fast fallback if time is critical ---
        if game_state.game_clock < 3.0:
            if CheckAction in legal_actions:
                return CheckAction()
            if CallAction in legal_actions and continue_cost <= BIG_BLIND:
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CheckAction()

        # =================================================================
        # PRE-FLOP (street == 0)
        # =================================================================
        if street == 0:
            return self._preflop(
                my_cards, legal_actions, continue_cost,
                my_pip, min_raise, max_raise, has_bounty, pot_size
            )

        # =================================================================
        # POST-FLOP (street >= 3)
        # =================================================================
        return self._postflop(
            my_cards, board_cards, legal_actions, continue_cost,
            my_pip, my_stack, min_raise, max_raise, has_bounty, pot_size
        )

    # -------------------------------------------------------------------
    # PRE-FLOP STRATEGY
    # -------------------------------------------------------------------

    def _preflop(self, my_cards, legal_actions, continue_cost,
                 my_pip, min_raise, max_raise, has_bounty, pot_size):
        """
        Pre-flop: RAISE premium hands, FOLD everything else.
        Bounty bonus: CALL (not raise) with hands containing the bounty rank.
        """
        canonical = _canonicalize_hand(my_cards[0], my_cards[1])
        is_premium = canonical in _PREMIUM_HANDS

        if is_premium:
            # --- Premium hand: RAISE aggressively ---
            if RaiseAction in legal_actions:
                # Size: 2.5x the pot or 3x the continue cost, whichever is larger
                target = max(min_raise, my_pip + max(6, continue_cost * 3))
                return RaiseAction(min(target, max_raise))
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()

        # --- Bounty rank in hand: widen range to CALL small bets ---
        if has_bounty:
            if continue_cost == 0 and CheckAction in legal_actions:
                return CheckAction()
            if continue_cost <= 4 and CallAction in legal_actions:
                return CallAction()

        # --- Junk hand: FOLD (or check if free) ---
        if continue_cost == 0 and CheckAction in legal_actions:
            return CheckAction()
        if FoldAction in legal_actions:
            return FoldAction()

        return CallAction()

    # -------------------------------------------------------------------
    # POST-FLOP STRATEGY
    # -------------------------------------------------------------------

    def _postflop(self, my_cards, board_cards, legal_actions, continue_cost,
                  my_pip, my_stack, min_raise, max_raise, has_bounty, pot_size):
        """
        Post-flop: eval7 hand evaluation → deterministic action.

        Strong (Two Pair+, Top Pair, Overpair) → RAISE
        Weak pair + free                       → CHECK (or BLUFF ~15%)
        Weak pair + facing bet                 → FOLD (or CALL if small)
        High card + free                       → CHECK (or BLUFF ~15%)
        High card + facing bet                 → FOLD
        """
        # Parse all cards into eval7 objects
        all_cards = [eval7.Card(c) for c in my_cards + board_cards]

        # Evaluate absolute hand strength
        hand_rank: int = eval7.evaluate(all_cards)
        hand_type: str = eval7.handtype(hand_rank)

        # --- DETERMINISTIC BLUFFING ---
        # Hash the visible game state to get a stable 0-99 int.
        # Same board + pot always produces the same decision (deterministic).
        bluff_factor = hash("".join(board_cards) + str(pot_size)) % 100

        # --- STRONG HANDS: Two Pair or better → RAISE ---
        if hand_type in _STRONG_HAND_TYPES:
            if RaiseAction in legal_actions:
                # Bet ~2/3 pot for value
                bet_size = max(min_raise, my_pip + (pot_size * 2 // 3))
                return RaiseAction(min(bet_size, max_raise))
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()

        # --- PAIR: check if top pair / overpair ---
        if hand_type == 'Pair':
            is_strong_pair = _is_top_pair_or_overpair(my_cards, board_cards)

            if is_strong_pair:
                # Top pair / overpair → RAISE for value
                if RaiseAction in legal_actions:
                    bet_size = max(min_raise, my_pip + (pot_size // 2))
                    return RaiseAction(min(bet_size, max_raise))
                if CallAction in legal_actions:
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
            else:
                # Weak pair — check if free, call small bets, fold large bets
                if continue_cost == 0 and CheckAction in legal_actions:
                    # ~15% of the time: bluff-raise with a weak pair
                    if bluff_factor < 15 and RaiseAction in legal_actions:
                        bet_size = max(min_raise, my_pip + (pot_size * 2 // 3))
                        return RaiseAction(min(bet_size, max_raise))
                    return CheckAction()
                # Call if bet is < 1/3 pot (pot odds make it worthwhile)
                if continue_cost <= max(pot_size // 3, 4) and CallAction in legal_actions:
                    return CallAction()
                # Bounty in hand: slightly wider calling range
                if has_bounty and continue_cost <= max(pot_size // 2, 6) and CallAction in legal_actions:
                    return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()

        # --- HIGH CARD: check if free, fold to any bet ---
        if continue_cost == 0 and CheckAction in legal_actions:
            # ~15% of the time: bluff-raise with nothing (pure bluff)
            if bluff_factor < 15 and RaiseAction in legal_actions:
                bet_size = max(min_raise, my_pip + (pot_size * 2 // 3))
                return RaiseAction(min(bet_size, max_raise))
            return CheckAction()
        if FoldAction in legal_actions:
            return FoldAction()

        # Fallback safety
        if CheckAction in legal_actions:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
