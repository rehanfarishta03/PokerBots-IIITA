'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot


class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.rank_value = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "T": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }
        self.strong_ranks = {"A", "K", "Q", "J", "T"}

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        pass

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        continue_cost = opp_pip - my_pip
        bankroll = game_state.bankroll
        game_clock = game_state.game_clock
        my_bounty = round_state.bounties[active]

        rank_a = my_cards[0][0]
        rank_b = my_cards[1][0]
        suit_a = my_cards[0][1]
        suit_b = my_cards[1][1]

        value_a = self.rank_value[rank_a]
        value_b = self.rank_value[rank_b]
        high = max(value_a, value_b)
        low = min(value_a, value_b)
        gap = high - low

        is_pair = rank_a == rank_b
        is_suited = suit_a == suit_b
        has_bounty_rank = rank_a == my_bounty or rank_b == my_bounty

        premium_preflop = (
            is_pair
            and high >= 10
        ) or (
            high >= 13 and low >= 10
        ) or (
            high == 14 and low >= 11
        ) or (
            is_suited and high >= 12 and gap <= 1
        )
        strong_preflop = premium_preflop or (
            is_pair
            or high >= 13
            or (high >= 11 and low >= 10)
            or (is_suited and gap <= 1 and high >= 10)
        )
        medium_preflop = high >= 12 or (is_suited and gap <= 2 and high >= 10) or low >= 9
        if has_bounty_rank:
            medium_preflop = True

        # Keep decisions simple when low on time to avoid clock losses.
        if game_clock < 5.0:
            if continue_cost == 0 and CheckAction in legal_actions:
                return CheckAction()
            if is_pair or high >= 12:
                if CallAction in legal_actions:
                    return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction()

        pot_size = (STARTING_STACK - round_state.stacks[active]) + (STARTING_STACK - round_state.stacks[1-active])
        fold_threshold = 8 if bankroll > 120 else 10 if bankroll > -120 else 12

        if street == 0:
            if continue_cost == 0:
                # Free option: apply pressure mostly with stronger ranges.
                if RaiseAction in legal_actions and (strong_preflop or (has_bounty_rank and high >= 10)):
                    min_raise, max_raise = round_state.raise_bounds()
                    target = max(min_raise, my_pip + 12)
                    if premium_preflop:
                        target = max(target, my_pip + 24)
                    return RaiseAction(min(target, max_raise))
                if CheckAction in legal_actions:
                    return CheckAction()

            if continue_cost > 0:
                # Exploit wide preflop callers: jam more often with premium hands.
                if RaiseAction in legal_actions and premium_preflop and my_stack > continue_cost:
                    _, max_raise = round_state.raise_bounds()
                    return RaiseAction(max_raise)

                if RaiseAction in legal_actions and (strong_preflop and continue_cost <= 10):
                    min_raise, max_raise = round_state.raise_bounds()
                    target = my_pip + max(10, continue_cost * 3)
                    return RaiseAction(min(max(target, min_raise), max_raise))

                if not medium_preflop and continue_cost >= max(fold_threshold, pot_size // 3) and FoldAction in legal_actions:
                    return FoldAction()

                if CallAction in legal_actions and (strong_preflop or continue_cost <= 3):
                    return CallAction()

                if FoldAction in legal_actions:
                    return FoldAction()

        if street > 0:
            if continue_cost == 0:
                # Value-heavy betting vs passive/cally lines.
                if RaiseAction in legal_actions and (premium_preflop or (strong_preflop and has_bounty_rank)):
                    min_raise, max_raise = round_state.raise_bounds()
                    bet_size = max(min_raise, pot_size // 2)
                    if bet_size <= max_raise and my_stack > 16:
                        return RaiseAction(bet_size)
                if CheckAction in legal_actions:
                    return CheckAction()

            if continue_cost > 0:
                # Avoid paying off with weak holdings.
                if CallAction in legal_actions and (
                    premium_preflop
                    or (strong_preflop and continue_cost <= max(4, pot_size // 5))
                    or (has_bounty_rank and continue_cost <= max(6, pot_size // 6))
                ):
                    return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()

        if CallAction in legal_actions:
            return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()
        if FoldAction in legal_actions:
            return FoldAction()
        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
