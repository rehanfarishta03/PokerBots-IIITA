#!/usr/bin/env python3
"""Deep CFR training scaffold for PokerBots-IIITA.

This file establishes the first local Deep CFR pipeline using the repository's engine
and state transition logic. It is intentionally a starting point rather than a
full production-grade solver.

Usage:
    python scripts/deep_cfr.py train --epochs 10 --trajectories 200 --save-dir deep_cfr_models

Requirements:
    python3, torch, eval7

"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import eval7
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from engine import (
    RoundState,
    TerminalState,
    FoldAction,
    CallAction,
    CheckAction,
    RaiseAction,
    STARTING_STACK,
    BIG_BLIND,
    SMALL_BLIND,
)

CARD_RANKS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
ACTION_NAMES = ["fold", "call", "raise_small", "raise_medium", "raise_max"]
NUM_ACTIONS = len(ACTION_NAMES)


def card_to_one_hot(card):
    rank_value = card.rank
    if isinstance(rank_value, int):
        rank_char = CARD_RANKS[rank_value]
    else:
        rank_char = str(rank_value)
    suit_value = card.suit
    if isinstance(suit_value, int):
        suit_char = SUIT_CHARS[suit_value]
    else:
        suit_char = str(suit_value)
    rank = CARD_RANKS.index(rank_char)
    suit = SUIT_CHARS.index(suit_char)
    one_hot = [0.0] * (len(CARD_RANKS) + len(SUIT_CHARS))
    one_hot[rank] = 1.0
    one_hot[len(CARD_RANKS) + suit] = 1.0
    return one_hot


def encode_round_state(round_state: RoundState, active: int) -> torch.Tensor:
    """Encode the current information set for the active player."""
    hole_cards = round_state.hands[active]
    board = round_state.deck.peek(round_state.street) if round_state.street > 0 else []
    board_features = []
    for card in board:
        board_features.extend(card_to_one_hot(card))
    while len(board_features) < 5 * (len(CARD_RANKS) + len(SUIT_CHARS)):
        board_features.extend([0.0] * (len(CARD_RANKS) + len(SUIT_CHARS)))

    hole_features = []
    for card in hole_cards:
        hole_features.extend(card_to_one_hot(card))

    my_stack = float(round_state.stacks[active]) / STARTING_STACK
    opp_stack = float(round_state.stacks[1 - active]) / STARTING_STACK
    my_pip = float(round_state.pips[active]) / STARTING_STACK
    opp_pip = float(round_state.pips[1 - active]) / STARTING_STACK
    street = float(round_state.street) / 5.0
    button = float(round_state.button % 2)
    bounty_rank = CARD_RANKS.index(round_state.bounties[active]) if round_state.bounties[active] in CARD_RANKS else -1
    bounty_one_hot = [0.0] * len(CARD_RANKS)
    if bounty_rank >= 0:
        bounty_one_hot[bounty_rank] = 1.0

    features = []
    features.extend(hole_features)
    features.extend(board_features)
    features.extend(bounty_one_hot)
    features.extend([my_stack, opp_stack, my_pip, opp_pip, street, button])
    return torch.tensor(features, dtype=torch.float32)


def legal_action_mask(round_state: RoundState, active: int) -> torch.Tensor:
    legal = round_state.legal_actions()
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    if FoldAction in legal:
        mask[0] = True
    if CheckAction in legal or CallAction in legal:
        mask[1] = True
    if RaiseAction in legal:
        mask[2] = True
        mask[3] = True
        mask[4] = True
    return mask


def action_from_index(round_state: RoundState, active: int, index: int):
    legal = round_state.legal_actions()
    if index == 0 and FoldAction in legal:
        return FoldAction()
    if index == 1:
        return CheckAction() if CheckAction in legal else CallAction()
    if index >= 2 and RaiseAction in legal:
        min_raise, max_raise = round_state.raise_bounds()
        if index == 2:
            amount = int(min_raise + 0.25 * (max_raise - min_raise))
        elif index == 3:
            amount = int(min_raise + 0.5 * (max_raise - min_raise))
        else:
            amount = int(max_raise)
        amount = max(min_raise, min(amount, max_raise))
        return RaiseAction(amount)
    if CheckAction in legal:
        return CheckAction()
    if CallAction in legal:
        return CallAction()
    return FoldAction()


def sample_action(policy: torch.Tensor, mask: torch.Tensor) -> int:
    masked = policy.clone()
    masked[~mask] = 0.0
    if masked.sum() <= 0:
        masked = mask.to(torch.float32)
    return torch.multinomial(masked / masked.sum(), num_samples=1).item()


class StrategyNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepCFRTrainer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.encoder_size = len(CARD_RANKS) * 2 + len(SUIT_CHARS) * 2 + 5 * (len(CARD_RANKS) + len(SUIT_CHARS)) + len(CARD_RANKS) + 6
        self.regret_net = StrategyNet(self.encoder_size).to(self.device)
        self.avg_net = StrategyNet(self.encoder_size).to(self.device)
        self.style_net = StyleNet(self.encoder_size).to(self.device)
        self.regret_opt = optim.Adam(self.regret_net.parameters(), lr=1e-3)
        self.avg_opt = optim.Adam(self.avg_net.parameters(), lr=1e-3)
        self.style_opt = optim.Adam(self.style_net.parameters(), lr=1e-3)
        self.regret_memory = []
        self.avg_memory = []
        self.style_memory = []

    def get_strategy(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.regret_net(obs.unsqueeze(0)).squeeze(0)
        logits[~mask] = -1e9
        positive = F.relu(logits)
        if positive.sum() <= 0.0:
            strategy = mask.to(torch.float32)
        else:
            strategy = positive
        return strategy / strategy.sum()

    def get_average_strategy(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.avg_net(obs.unsqueeze(0)).squeeze(0)
        logits[~mask] = -1e9
        probs = F.softmax(logits, dim=-1)
        probs[~mask] = 0.0
        if probs.sum() <= 0.0:
            probs = mask.to(torch.float32)
        return probs / probs.sum()

    def traverse(self, round_state: RoundState, player: int, use_avg_policy: bool = False):
        if isinstance(round_state, TerminalState):
            return float(round_state.deltas[player])

        active = round_state.button % 2
        obs = encode_round_state(round_state, active).to(self.device)
        mask = legal_action_mask(round_state, active).to(self.device)
        if active == player:
            strategy = self.get_strategy(obs, mask)
            self.avg_memory.append((obs.cpu(), strategy.detach().cpu(), mask.cpu()))
            utilities = []
            for index in range(NUM_ACTIONS):
                if not mask[index]:
                    utilities.append(0.0)
                    continue
                action = action_from_index(round_state, active, index)
                next_state = round_state.proceed(action)
                utilities.append(self.traverse(next_state, player, use_avg_policy))
            node_value = sum(strategy[index] * utilities[index] for index in range(NUM_ACTIONS))
            regrets = [utilities[index] - node_value for index in range(NUM_ACTIONS)]
            self.regret_memory.append((obs.cpu(), torch.tensor(regrets, dtype=torch.float32), mask.cpu()))
            return node_value

        policy = self.get_average_strategy(obs, mask) if use_avg_policy else self.get_strategy(obs, mask)
        index = sample_action(policy, mask)
        action = action_from_index(round_state, active, index)
        next_state = round_state.proceed(action)
        return self.traverse(next_state, player, use_avg_policy)

    def train_iteration(self, iterations: int = 200):
        for _ in range(iterations):
            root_state = self._sample_initial_round()
            player = random.choice([0, 1])
            self.traverse(root_state, player)

        self._fit_regret_net(batch_size=32, epochs=2)
        self._fit_average_net(batch_size=32, epochs=2)
        self.regret_memory.clear()
        self.avg_memory.clear()
        self._generate_style_data(max(10, iterations // 2))
        self._fit_style_net(batch_size=32, epochs=2)
        self.style_memory.clear()

    def _fit_regret_net(self, batch_size: int, epochs: int):
        if not self.regret_memory:
            return
        dataset = self.regret_memory
        for _ in range(epochs):
            random.shuffle(dataset)
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start : start + batch_size]
                obs_batch = torch.stack([item[0] for item in batch]).to(self.device)
                reg_batch = torch.stack([item[1] for item in batch]).to(self.device)
                mask_batch = torch.stack([item[2] for item in batch]).to(self.device)
                preds = self.regret_net(obs_batch)
                preds = preds * mask_batch.to(preds.dtype)
                loss = F.mse_loss(preds, reg_batch)
                self.regret_opt.zero_grad()
                loss.backward()
                self.regret_opt.step()

    def _fit_average_net(self, batch_size: int, epochs: int):
        if not self.avg_memory:
            return
        dataset = self.avg_memory
        for _ in range(epochs):
            random.shuffle(dataset)
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start : start + batch_size]
                obs_batch = torch.stack([item[0] for item in batch]).to(self.device)
                strat_batch = torch.stack([item[1] for item in batch]).to(self.device)
                mask_batch = torch.stack([item[2] for item in batch]).to(self.device)
                preds = self.avg_net(obs_batch)
                preds = preds.masked_fill(~mask_batch, -1e9)
                loss = F.mse_loss(F.softmax(preds, dim=-1), strat_batch)
                self.avg_opt.zero_grad()
                loss.backward()
                self.avg_opt.step()

    def get_style(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.style_net(obs.unsqueeze(0)).squeeze(0)
        return F.softmax(logits, dim=-1)

    def _fit_style_net(self, batch_size: int, epochs: int):
        if not self.style_memory:
            return
        dataset = self.style_memory
        for _ in range(epochs):
            random.shuffle(dataset)
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start : start + batch_size]
                obs_batch = torch.stack([item[0] for item in batch]).to(self.device)
                label_batch = torch.tensor([item[1] for item in batch], dtype=torch.long, device=self.device)
                preds = self.style_net(obs_batch)
                loss = F.cross_entropy(preds, label_batch)
                self.style_opt.zero_grad()
                loss.backward()
                self.style_opt.step()

    def _scripted_style_policy(self, round_state: RoundState, active: int, aggressive: bool):
        legal = round_state.legal_actions()
        continue_cost = round_state.pips[1-active] - round_state.pips[active]

        if aggressive:
            if RaiseAction in legal and random.random() < 0.85:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(max_raise)
            if CallAction in legal and continue_cost > 0:
                return CallAction()
            if CheckAction in legal:
                return CheckAction()
            return FoldAction()

        if continue_cost == 0:
            if CheckAction in legal:
                return CheckAction()
            if RaiseAction in legal and random.random() < 0.1:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(min(min_raise + 2, max_raise))
            return FoldAction() if FoldAction in legal else CallAction()

        if CallAction in legal and continue_cost <= 4:
            return CallAction()
        if FoldAction in legal:
            return FoldAction()
        if CheckAction in legal:
            return CheckAction()
        return CallAction()

    def _generate_style_data(self, trajectories: int = 50):
        for _ in range(trajectories):
            aggressive = random.choice([0, 1])
            root_state = self._sample_initial_round()
            while not isinstance(root_state, TerminalState):
                active = root_state.button % 2
                if active == 0:
                    obs = encode_round_state(root_state, active).to(self.device)
                    self.style_memory.append((obs.cpu(), aggressive))
                    action = self._policy_action(root_state, active, use_avg_policy=True)
                else:
                    action = self._scripted_style_policy(root_state, active, aggressive=bool(aggressive))
                root_state = root_state.proceed(action)

    def evaluate_style(self, games: int = 100, use_avg_policy: bool = True) -> dict[str, float]:
        correct = 0
        total = 0
        for _ in range(games):
            aggressive = random.choice([0, 1])
            root_state = self._sample_initial_round()
            while not isinstance(root_state, TerminalState):
                active = root_state.button % 2
                if active == 0:
                    obs = encode_round_state(root_state, active).to(self.device)
                    style_probs = self.get_style(obs)
                    predicted = int(style_probs.argmax().item())
                    correct += int(predicted == aggressive)
                    total += 1
                    action = self._policy_action(root_state, active, use_avg_policy=use_avg_policy)
                else:
                    action = self._scripted_style_policy(root_state, active, aggressive=bool(aggressive))
                root_state = root_state.proceed(action)
        return {
            "games": games,
            "observations": total,
            "accuracy": correct / total if total else 0.0,
        }

    def _policy_action(self, round_state: RoundState, active: int, use_avg_policy: bool = False):
        obs = encode_round_state(round_state, active).to(self.device)
        mask = legal_action_mask(round_state, active).to(self.device)
        if use_avg_policy:
            policy = self.get_average_strategy(obs, mask)
        else:
            policy = self.get_strategy(obs, mask)
        index = sample_action(policy, mask)
        return action_from_index(round_state, active, index)

    def _play_game(self, round_state: RoundState, use_avg_policy: bool = False) -> float:
        while not isinstance(round_state, TerminalState):
            active = round_state.button % 2
            action = self._policy_action(round_state, active, use_avg_policy)
            round_state = round_state.proceed(action)
        return float(round_state.deltas[0])

    def evaluate(self, games: int = 100, use_avg_policy: bool = True) -> dict[str, float]:
        total_delta = 0.0
        wins = 0
        losses = 0
        draws = 0
        for _ in range(games):
            root_state = self._sample_initial_round()
            result = self._play_game(root_state, use_avg_policy=use_avg_policy)
            total_delta += result
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1
        return {
            "games": games,
            "total_delta": total_delta,
            "avg_delta": total_delta / games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }

    def _sample_initial_round(self) -> RoundState:
        # Use engine semantics: create a shuffled deck and deal two cards to each player.
        deck_obj = eval7.Deck()
        deck_obj.shuffle()
        hands = [deck_obj.deal(2), deck_obj.deal(2)]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        bounties = [random.choice(CARD_RANKS), random.choice(CARD_RANKS)]
        return RoundState(0, 0, pips, stacks, hands, deck_obj, bounties, None)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.regret_net.state_dict(), path / "regret_net.pt")
        torch.save(self.avg_net.state_dict(), path / "avg_net.pt")
        torch.save(self.style_net.state_dict(), path / "style_net.pt")

    def load(self, path: Path):
        self.regret_net.load_state_dict(torch.load(path / "regret_net.pt", map_location=self.device))
        self.avg_net.load_state_dict(torch.load(path / "avg_net.pt", map_location=self.device))
        self.style_net.load_state_dict(torch.load(path / "style_net.pt", map_location=self.device))


def main():
    parser = argparse.ArgumentParser(description="Deep CFR training entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a Deep CFR model")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--trajectories", type=int, default=200)
    train_parser.add_argument("--save-dir", default="deep_cfr_models")
    train_parser.add_argument("--device", default="cpu")

    eval_parser = subparsers.add_parser("eval", help="Evaluate learned policy against random baseline")
    eval_parser.add_argument("--model-dir", default="deep_cfr_models")
    eval_parser.add_argument("--games", type=int, default=100)
    eval_parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.command == "train":
        trainer = DeepCFRTrainer(device=args.device)
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}: sampling {args.trajectories} trajectories")
            trainer.train_iteration(iterations=args.trajectories)
            trainer.save(Path(args.save_dir))
            print(f"Saved models to {args.save_dir}")

    elif args.command == "eval":
        trainer = DeepCFRTrainer(device=args.device)
        trainer.load(Path(args.model_dir))
        print(f"Loaded model from {args.model_dir}")
        stats = trainer.evaluate(games=args.games, use_avg_policy=True)
        style_stats = trainer.evaluate_style(games=args.games, use_avg_policy=True)
        print("Evaluation results:")
        print(f"  games: {stats['games']}")
        print(f"  total delta: {stats['total_delta']:.2f}")
        print(f"  average delta: {stats['avg_delta']:.4f}")
        print(f"  wins: {stats['wins']}")
        print(f"  losses: {stats['losses']}")
        print(f"  draws: {stats['draws']}")
        print("Style classification results:")
        print(f"  observations: {style_stats['observations']}")
        print(f"  accuracy: {style_stats['accuracy']:.4f}")


if __name__ == "__main__":
    main()
