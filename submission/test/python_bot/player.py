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
        self.tight_preflop_ranks = {"A", "K", "Q", "J", "T"}

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
        continue_cost = opp_pip - my_pip

        # Prefer checking when free.
        if continue_cost == 0 and CheckAction in legal_actions:
            return CheckAction()

        # Fold to very large pressure with weak preflop holdings.
        pot_size = (STARTING_STACK - round_state.stacks[active]) + (STARTING_STACK - round_state.stacks[1-active])
        high_pressure = continue_cost > max(8, pot_size // 2)

        rank_a = my_cards[0][0]
        rank_b = my_cards[1][0]
        is_pair = rank_a == rank_b
        strong_preflop = is_pair or (rank_a in self.tight_preflop_ranks and rank_b in self.tight_preflop_ranks)

        if street == 0 and high_pressure and not strong_preflop and FoldAction in legal_actions:
            return FoldAction()

        # Occasionally value-bet small with strong preflop hands when raising is available.
        if street == 0 and strong_preflop and RaiseAction in legal_actions and my_stack > continue_cost:
            min_raise, _ = round_state.raise_bounds()
            return RaiseAction(min_raise)

        if CallAction in legal_actions:
            return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()
        if FoldAction in legal_actions:
            return FoldAction()
        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
