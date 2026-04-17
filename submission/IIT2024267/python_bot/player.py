'''
Strong Poker Bot — Clean rewrite.

Strategy:
  1. Preflop: Tight-aggressive hand chart (raise strong, call medium, fold weak).
  2. Post-flop: Monte-Carlo equity + pot-odds for every decision.
  3. Bounty: Bump aggression when our bounty rank is live.
  4. CFR-style regret table, trained offline via self-play and loaded at start.
     Falls back to pure equity if no table is found (still beats a simple bot).

The CFR training is a standalone function you run ONCE from the terminal:
    python player.py --train --iters 20000

That writes  cfr_regrets.json  next to player.py.
During the match the file is loaded and used for look-ups; if it is missing
the bot falls back to the equity-only strategy which is already strong.

Time budget per action: <30 ms  (equity MC uses 200 samples w/ eval7).
'''

import random, math, os, json, argparse, time, itertools
from collections import defaultdict

try:
    import eval7
    _E7 = True
except ImportError:
    _E7 = False

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states  import GameState, TerminalState, RoundState
from skeleton.states  import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot     import Bot
from skeleton.runner  import parse_args, run_bot

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RANKS       = '23456789TJQKA'
SUITS       = 'shdc'
RANK_VAL    = {r: i for i, r in enumerate(RANKS)}
ALL_CARDS   = [r + s for r in RANKS for s in SUITS]   # 52 strings
REGRET_FILE = os.path.join(os.path.dirname(__file__), 'cfr_regrets.json')

# ─────────────────────────────────────────────────────────────────────────────
# EQUITY  (fast Monte-Carlo using eval7 when available)
# ─────────────────────────────────────────────────────────────────────────────

def equity(hole: list, board: list, samples: int = 200) -> float:
    '''Win probability for hole cards given current board. [0,1]'''
    if len(hole) < 2:
        return 0.5
    known   = set(hole + board)
    deck    = [c for c in ALL_CARDS if c not in known]
    need    = 5 - len(board)
    wins = ties = 0

    if _E7 and eval7 is not None:
        h_e7 = [eval7.Card(c) for c in hole]
        b_e7 = [eval7.Card(c) for c in board]
        for _ in range(samples):
            draw      = random.sample(deck, 2 + need)
            opp_e7    = [eval7.Card(c) for c in draw[:2]]
            extra_e7  = [eval7.Card(c) for c in draw[2:]]
            full      = b_e7 + extra_e7
            me        = eval7.evaluate(h_e7  + full)
            opp       = eval7.evaluate(opp_e7 + full)
            if me > opp:   wins += 1
            elif me == opp: ties += 1
    else:
        for _ in range(samples):
            draw      = random.sample(deck, 2 + need)
            opp_hole  = draw[:2]
            full      = board + draw[2:]
            me        = _score_hand(hole,     full)
            opp       = _score_hand(opp_hole, full)
            if me > opp:   wins += 1
            elif me == opp: ties += 1

    return (wins + 0.5 * ties) / samples


def _score_hand(hole, board):
    cards = hole + board
    best  = -1
    for combo in itertools.combinations(cards, 5):
        s = _score5(combo)
        if s > best: best = s
    return best

def _score5(cards):
    rv  = sorted([RANK_VAL[c[0]] for c in cards], reverse=True)
    flu = len({c[1] for c in cards}) == 1
    st  = (rv == list(range(rv[0], rv[0]-5, -1))) or rv == [12,3,2,1,0]
    rc  = defaultdict(int)
    for r in rv: rc[r] += 1
    freq = sorted(rc.values(), reverse=True)
    dist = sorted(rc, key=lambda r:(rc[r],r), reverse=True)
    tb   = sum(r*(13**i) for i,r in enumerate(reversed(dist)))
    if st and flu: cat=8
    elif freq[0]==4: cat=7
    elif freq[:2]==[3,2]: cat=6
    elif flu: cat=5
    elif st: cat=4
    elif freq[0]==3: cat=3
    elif freq[:2]==[2,2]: cat=2
    elif freq[0]==2: cat=1
    else: cat=0
    return cat*(13**6)+tb


def pot_odds(call_cost: int, pot: int) -> float:
    '''Minimum equity needed to call profitably.'''
    total = pot + call_cost
    if total <= 0: return 0.0
    return call_cost / total


# ─────────────────────────────────────────────────────────────────────────────
# BOARD EXTRACTION  (handles both eval7.Deck and plain lists)
# ─────────────────────────────────────────────────────────────────────────────

def board_cards(rs: RoundState) -> list:
    s = rs.street
    if s == 0: return []
    d = rs.deck
    if d is None: return []
    if isinstance(d, list):
        return [str(c) for c in d[:s]]
    try:
        return [str(c) for c in d.peek(s)]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# PREFLOP HAND STRENGTH  (no equity MC needed — pure chart)
# ─────────────────────────────────────────────────────────────────────────────

def preflop_strength(hole: list) -> float:
    '''
    Returns a [0,1] hand-strength score based on Chen formula
    approximation. No randomness — deterministic and instant.
    '''
    if len(hole) < 2: return 0.0
    r0, r1 = hole[0][0], hole[1][0]
    s0, s1 = hole[0][1], hole[1][1]
    v0, v1 = RANK_VAL[r0], RANK_VAL[r1]
    if v0 < v1: v0, v1 = v1, v0          # v0 ≥ v1

    suited = (s0 == s1)
    gap    = v0 - v1

    # Chen formula (simplified, scaled to [0,1])
    score = 0.0

    # High card value
    score += v0 / 2.0

    # Pair bonus
    if gap == 0:
        score = max(score, 5.0)
        if v0 >= 10: score += 4
        elif v0 >= 6: score += 2
        else:         score += 1

    # Suited bonus
    if suited: score += 2

    # Connector bonus
    if   gap == 0: pass           # pair already handled
    elif gap == 1: score += 1
    elif gap == 2: score += 0.5

    # Penalty for gaps
    if   gap == 3: score -= 1
    elif gap == 4: score -= 2
    elif gap >= 5: score -= 4

    # Normalise roughly to [0,1];  max Chen score ~20 for AA
    return min(max(score / 20.0, 0.0), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# CFR  (Vanilla CFR over a small abstraction)
# ─────────────────────────────────────────────────────────────────────────────
# Abstraction:
#   • Preflop: 10 equity buckets of preflop_strength
#   • Post-flop: 10 equity buckets via MC equity
#   • Betting: check/fold/call/raise-small/raise-big  (5 abstract actions max)
#   • History: last 3 abstract actions only (keeps key space tiny)
# ─────────────────────────────────────────────────────────────────────────────

N_BUCKETS = 10      # equity buckets
MAX_DEPTH = 12      # recursion guard

A_FOLD  = 'f'
A_CHECK = 'k'
A_CALL  = 'c'
A_RAISE = 'r'    # moderate raise (~60% pot)
A_ALLIN = 'a'    # all-in / max raise

# Actions that end a street when both players have acted
_PASSIVE = {A_CHECK, A_CALL}
_STREETS = [0, 3, 4, 5]    # preflop, flop, turn, river


def _legal_abstract(last_opp: str | None, n_raises: int) -> list:
    '''Abstract legal actions given last opponent action and raise count.'''
    if last_opp in (A_RAISE, A_ALLIN):
        acts = [A_FOLD, A_CALL]
        if n_raises < 2:
            acts += [A_RAISE, A_ALLIN]
        return acts
    else:   # check / None (first to act)
        acts = [A_CHECK]
        if n_raises < 2:
            acts += [A_RAISE, A_ALLIN]
        return acts


def _street_over(hist: list) -> bool:
    '''True when the current street should advance.'''
    if len(hist) < 2: return False
    return hist[-1] in _PASSIVE and hist[-2] in _PASSIVE


def _next_street(street: int) -> int:
    idx = _STREETS.index(street) if street in _STREETS else -1
    if idx == -1 or idx == len(_STREETS) - 1:
        return 99   # showdown
    return _STREETS[idx + 1]


def _eq_bucket(hs: float) -> int:
    return min(int(hs * N_BUCKETS), N_BUCKETS - 1)


class _Node:
    __slots__ = ('r', 's')
    def __init__(self, n):
        self.r = [0.0] * n   # regret sums
        self.s = [0.0] * n   # strategy sums

    def strategy(self, reach: float) -> list:
        pos = [max(x, 0.0) for x in self.r]
        tot = sum(pos)
        if tot > 1e-12:
            sig = [p / tot for p in pos]
        else:
            sig = [1.0 / len(self.r)] * len(self.r)
        for i, v in enumerate(sig):
            self.s[i] += reach * v
        return sig

    def avg(self) -> list:
        tot = sum(self.s)
        if tot > 1e-12:
            return [v / tot for v in self.s]
        return [1.0 / len(self.s)] * len(self.s)


class CFR:
    def __init__(self):
        self._nodes: dict = {}

    def _key(self, bucket: int, street: int, hist: list) -> str:
        h = ''.join(hist[-3:])
        return f'{bucket},{street},{h}'

    def _node(self, key: str, n: int) -> _Node:
        if key not in self._nodes:
            self._nodes[key] = _Node(n)
        elif self._nodes[key].r.__len__() != n:
            self._nodes[key] = _Node(n)
        return self._nodes[key]

    def _terminal_util(self, street: int, eq0: float,
                       pot: int, active: int) -> tuple:
        '''Return (util0, util1) at a terminal node.'''
        u0 = eq0 * pot - (1 - eq0) * pot
        return (u0, -u0)

    def _cfr(self, eq0: float, eq1: float, street: int,
             hist: list, p0: float, p1: float,
             pot: int, depth: int) -> tuple:

        if depth > MAX_DEPTH:
            u = eq0 * pot - (1 - eq0) * pot
            return (u, -u)

        # Fold terminal
        if hist and hist[-1] == A_FOLD:
            # last actor folded; the one before wins
            last_actor = (depth - 1) % 2
            if last_actor == 0:
                return (-pot * 0.5, pot * 0.5)
            else:
                return (pot * 0.5, -pot * 0.5)

        # Showdown
        if street == 99:
            u = eq0 * pot - (1 - eq0) * pot
            return (u, -u)

        # Street advance
        if _street_over(hist):
            return self._cfr(eq0, eq1, _next_street(street),
                             [], p0, p1, pot, depth)

        active  = depth % 2
        eq_act  = eq0 if active == 0 else eq1
        bucket  = _eq_bucket(eq_act)

        last_opp_idx = len(hist) - 1 if hist else -1
        last_opp = hist[-1] if hist else None
        n_raises = sum(1 for a in hist if a in (A_RAISE, A_ALLIN))

        acts    = _legal_abstract(last_opp, n_raises)
        key     = self._key(bucket, street, hist)
        node    = self._node(key, len(acts))
        rw      = p0 if active == 0 else p1
        sigma   = node.strategy(rw)

        child_utils = []
        for i, a in enumerate(acts):
            new_hist = hist + [a]
            new_pot  = pot
            if a == A_CALL:
                new_pot += min(20, pot // 4)     # rough call cost
            elif a == A_RAISE:
                new_pot += max(2, pot // 2)
            elif a == A_ALLIN:
                new_pot += pot * 2

            np0 = p0 * sigma[i] if active == 0 else p0
            np1 = p1 * sigma[i] if active == 1 else p1

            u = self._cfr(eq0, eq1, street, new_hist,
                          np0, np1, new_pot, depth + 1)
            child_utils.append(u)

        # Node utility
        u0 = sum(sigma[i] * child_utils[i][0] for i in range(len(acts)))
        u1 = sum(sigma[i] * child_utils[i][1] for i in range(len(acts)))

        opp_reach = p1 if active == 0 else p0
        node_u = u0 if active == 0 else u1

        for i in range(len(acts)):
            cf = child_utils[i][0] if active == 0 else child_utils[i][1]
            node.r[i] += opp_reach * (cf - node_u)

        return (u0, u1)

    def train(self, iters: int = 10000, verbose: bool = True):
        t0 = time.time()
        for i in range(iters):
            eq0 = random.random()
            eq1 = random.random()
            pot = BIG_BLIND + SMALL_BLIND
            self._cfr(eq0, eq1, 0, [], 1.0, 1.0, pot, 0)
            if verbose and (i+1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f'  iter {i+1}/{iters}   nodes={len(self._nodes)}  '
                      f't={elapsed:.1f}s')
        print(f'Done. {len(self._nodes)} nodes.')

    def get_avg_strategy(self, bucket: int, street: int,
                         hist: list) -> dict | None:
        '''Returns {action: prob} or None if key not found.'''
        key  = self._key(bucket, street, hist)
        if key not in self._nodes:
            return None
        node = self._nodes[key]
        last_opp = hist[-1] if hist else None
        n_raises = sum(1 for a in hist if a in (A_RAISE, A_ALLIN))
        acts = _legal_abstract(last_opp, n_raises)
        avg  = node.avg()
        return {a: avg[i] for i, a in enumerate(acts)}

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = REGRET_FILE):
        data = {}
        for k, node in self._nodes.items():
            data[k] = {'r': node.r, 's': node.s}
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f'Saved {len(data)} CFR nodes → {path}')

    def load(self, path: str = REGRET_FILE) -> bool:
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            n = _Node(len(v['r']))
            n.r = v['r']
            n.s = v['s']
            self._nodes[k] = n
        print(f'Loaded {len(self._nodes)} CFR nodes from {path}')
        return True


# ─────────────────────────────────────────────────────────────────────────────
# BOUNTY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bounty_active(hole: list, board: list, rank: str) -> bool:
    return any(c[0] == rank for c in hole + board)

def bounty_possible(hole: list, rank: str) -> bool:
    return any(c[0] == rank for c in hole)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER
# ─────────────────────────────────────────────────────────────────────────────

class Player(Bot):
    '''
    Equity-first poker bot with optional CFR strategy overlay.

    Decision flow:
      1. Compute equity (preflop: Chen score, post-flop: MC).
      2. If a trained CFR node exists for this info-set, sample from it.
      3. Otherwise fall back to direct equity decision rules.
      4. Bounty rank present → multiply raise size.
    '''

    # Equity thresholds for direct rules (post-flop)
    RAISE_THRESH = 0.72  # raise if equity > this
    CALL_THRESH  = 0.50   # call  if equity > this (else fold)

    # Preflop thresholds (Chen score, normalised)
    PRE_RAISE    = 0.70
    PRE_CALL     = 0.45

    def __init__(self):
        self.cfr = CFR()
        self._loaded = self.cfr.load()
        self._hist: list = []   # abstract action history this hand
        self._hole: list = []
        self._bounty: str = ''

    # ── lifecycle ────────────────────────────────────────────────────────────

    def handle_new_round(self, gs: GameState, rs: RoundState, active: int):
        self._hist   = []
        self._hole   = rs.hands[active] if rs.hands[active] else []
        self._bounty = rs.bounties[active] if rs.bounties[active] else ''

    def handle_round_over(self, gs: GameState, ts: TerminalState, active: int):
        pass   # online learning not needed once CFR is pre-trained

    # ── action ───────────────────────────────────────────────────────────────

    def get_action(self, gs: GameState, rs: RoundState, active: int):
        legal     = rs.legal_actions()
        street    = rs.street
        hole      = rs.hands[active]
        board     = board_cards(rs)
        my_pip    = rs.pips[active]
        opp_pip   = rs.pips[1 - active]
        my_stack  = rs.stacks[active]
        opp_stack = rs.stacks[1 - active]
        call_cost = opp_pip - my_pip
        pot       = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)
        bounty    = rs.bounties[active] if rs.bounties[active] else ''
        is_bb     = bool(active)

        if RaiseAction in legal:
            min_r, max_r = rs.raise_bounds()
        else:
            min_r = max_r = opp_pip

        # ── 1. Compute equity ────────────────────────────────────────────────
        if street == 0:
            eq = preflop_strength(hole)
        else:
            eq = equity(hole, board)
        eq-=0.03

        if not is_bb:
            eq -= 0.04
        else:
            eq += 0.02
        # ── 2. Bounty boost ──────────────────────────────────────────────────
        b_active   = bounty and bounty_active(hole, board, bounty)
        b_possible = bounty and bounty_possible(hole, bounty)
        if b_active:
            eq = min(eq + 0.12, 1.0)
        elif b_possible:
            eq = min(eq + 0.05, 1.0)

        # ── 3. CFR strategy look-up ──────────────────────────────────────────
        bucket = _eq_bucket(eq)
        cfr_strat = None
        if cfr_strat:
            action = self._cfr_action(cfr_strat, legal, pot, call_cost,
                                      min_r, max_r, my_pip, eq, street)
        else:
            action = self._equity_action(eq, legal, pot, call_cost,
                                         min_r, max_r, my_pip, street,
                                         is_bb, bounty)

        # ── 4. Record abstract action ────────────────────────────────────────
        self._hist.append(self._to_abstract(action, min_r, max_r))
        return action

    # ── helpers ──────────────────────────────────────────────────────────────

    def _cfr_action(self, strat: dict, legal: set, pot: int, call_cost: int,
                    min_r: int, max_r: int, my_pip: int,
                    eq: float, street: int):
        '''Sample from CFR strategy, then translate to engine action.'''
        # Threshold-prune (Chintamaneni §4)
        thresh = 0.15 if street == 0 else (0.20 if street == 3 else 0.25)
        total  = sum(strat.values())
        valid  = {a: p for a, p in strat.items()
                  if (p / max(total, 1e-9)) >= thresh}
        if not valid:
            valid = strat

        # Renormalise
        vtot = sum(valid.values())
        r    = random.random()
        cum  = 0.0
        chosen = A_CHECK
        for a, p in valid.items():
            cum += p / vtot
            if r < cum:
                chosen = a
                break

        return self._abstract_to_action(chosen, legal, pot, min_r, max_r, my_pip)

    def _equity_action(self, eq: float, legal: set, pot: int, call_cost: int,
                       min_r: int, max_r: int, my_pip: int,
                       street: int, is_bb: bool, bounty: str):
        '''Pure equity decision rule — fallback when CFR has no node.'''
        po = pot_odds(call_cost, pot)

        if call_cost > pot * 0.8 and eq < 0.60:
            return FoldAction()
        # Preflop
        if street == 0:
            if eq >= self.PRE_RAISE:
                if RaiseAction in legal:
                    # size: 2.5x BB normally, bigger with strong hands
                    size = int(pot * (0.6))
                    amt  = max(min_r, min(size, max_r))
                    return RaiseAction(amt)
                return CallAction() if CallAction in legal else CheckAction()

            elif eq >= self.PRE_CALL:
                if call_cost == 0:
                    return CheckAction() if CheckAction in legal else CallAction()
                return CallAction() if CallAction in legal else CheckAction()

            else:   # weak hand
                if is_bb and call_cost == 0 and CheckAction in legal:
                    return CheckAction()   # free look
                if call_cost == 0:
                    return CheckAction() if CheckAction in legal else FoldAction()
                return FoldAction() if FoldAction in legal else CheckAction()

        # Post-flop
        if eq >= self.RAISE_THRESH:
            if RaiseAction in legal:
                # Value bet: 55–80% pot
                frac = 0.55 + (eq - self.RAISE_THRESH) * 1.5
                size = int(pot * min(frac, 0.80)) + my_pip
                amt  = max(min_r, min(size, max_r))
                return RaiseAction(amt)
            return CallAction() if CallAction in legal else CheckAction()

        elif eq >= self.CALL_THRESH and eq >= po+0.06:
            if call_cost == 0:
                return CheckAction() if CheckAction in legal else CallAction()
            return CallAction() if CallAction in legal else CheckAction()

        else:
            if call_cost == 0:
                return CheckAction() if CheckAction in legal else FoldAction()
            # Occasional bluff (~8% of the time) when equity is close
            if eq > 0.40 and random.random() < 0.02 and RaiseAction in legal:
                return RaiseAction(min_r)
            return FoldAction() if FoldAction in legal else CheckAction()

    def _abstract_to_action(self, a: str, legal: set, pot: int,
                             min_r: int, max_r: int, my_pip: int):
        if a == A_FOLD:
            return FoldAction() if FoldAction in legal else \
                   (CheckAction() if CheckAction in legal else CallAction())
        if a == A_CHECK:
            if CheckAction in legal: return CheckAction()
            if CallAction  in legal: return CallAction()
            return FoldAction()
        if a == A_CALL:
            return CallAction() if CallAction in legal else \
                   (CheckAction() if CheckAction in legal else FoldAction())
        if a == A_RAISE:
            if RaiseAction in legal:
                size = int(pot * 0.60) + my_pip
                amt  = max(min_r, min(size, max_r))
                return RaiseAction(amt)
            return CallAction() if CallAction in legal else CheckAction()
        if a == A_ALLIN:
            if RaiseAction in legal:
                return RaiseAction(max_r)
            return CallAction() if CallAction in legal else CheckAction()
        return CheckAction() if CheckAction in legal else CallAction()

    def _to_abstract(self, action, min_r: int, max_r: int) -> str:
        if isinstance(action, FoldAction):  return A_FOLD
        if isinstance(action, CheckAction): return A_CHECK
        if isinstance(action, CallAction):  return A_CALL
        if isinstance(action, RaiseAction):
            mid = (min_r + max_r) // 2
            return A_ALLIN if action.amount >= mid else A_RAISE
        return A_CHECK


# ─────────────────────────────────────────────────────────────────────────────
# CLI  — standalone training
# ─────────────────────────────────────────────────────────────────────────────

def _train_and_save(iters: int):
    print(f'Training CFR for {iters} iterations …')
    cfr = CFR()
    cfr.train(iters, verbose=True)
    cfr.save(REGRET_FILE)
    print(f'Saved to {REGRET_FILE}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train',  action='store_true', help='Run CFR training then exit')
    ap.add_argument('--iters',  type=int, default=20000, help='Training iterations')
    ap.add_argument('--host',   type=str, default='localhost')
    ap.add_argument('port',     type=int, nargs='?', default=None)
    args = ap.parse_args()

    if args.train:
        _train_and_save(args.iters)
    else:
        if args.port is None:
            ap.print_help()
        else:
            # Re-parse with skeleton's parser for the actual bot run
            from skeleton.runner import parse_args as _pa, run_bot as _rb
            _rb(Player(), _pa())
