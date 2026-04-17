from submission.IIT2024011.python_bot.skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from submission.IIT2024011.python_bot.skeleton.states import GameState, TerminalState, RoundState
from submission.IIT2024011.python_bot.skeleton.bot import Bot
from submission.IIT2024011.python_bot.skeleton.runner import parse_args, run_bot

class Player(Bot):
    def __init__(self):
        """
        Called when a new game starts. 
        """
        pass

    def handle_new_round(self, game_state: GameState, round_state: RoundState, active: int):
        """
        Called when a new round starts. Called NUM_ROUNDS times.
        """
        pass

    def handle_round_over(self, game_state: GameState, terminal_state: TerminalState, active: int):
        """
        Called when a round ends. Called NUM_ROUNDS times.
        """
        pass

    def get_action(self, game_state: GameState, round_state: RoundState, active: int):
        """
        Always pushes the maximum amount of chips into the pot.
        """
        legal_actions = round_state.legal_actions()

        # 1. If we are allowed to raise, go all-in by raising the maximum bounds.
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            return RaiseAction(max_raise)
        
        # 2. If raising isn't allowed (e.g., opponent went all-in first or cap reached), we call.
        if CallAction in legal_actions:
            return CallAction()
            
        # 3. If we can't raise or call, checking is our only way to stay in the hand.
        if CheckAction in legal_actions:
            return CheckAction()
            
        # 4. Fallback (should theoretically never be reached in heads-up no-limit if we want to play)
        return FoldAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())