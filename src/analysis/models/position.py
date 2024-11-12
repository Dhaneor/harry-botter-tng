#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 10 22:18:20 2024

@author dhaneor
"""
import pandas as pd
import numpy as np


class Position:
    def __init__(self, symbol: str, position_type: str, df: pd.DataFrame):
        self.symbol = symbol
        self.position_type = position_type.upper()
        self.df = df

    def __repr__(self):
        return (
            f"Position(symbol='{self.symbol}', position_type='{self.position_type} :: "
            f"pnl={self.pnl:.3f}\t:: max_drawdown={self.max_drawdown:.4f}, :: "
            f"duration={self.duration} :: entry_price={self.entry_price} :: "
            f"exit_price={self.exit_price}')"
            )

    @property
    def is_long(self) -> bool:
        return self.position_type in (1, "LONG")

    @property
    def is_short(self) -> bool:
        return self.position_type in (2, "SHORT")

    @property
    def pnl(self) -> float:
        return self.df['b.value'].iloc[-1] - self.df['b.value'].iloc[0]

    @property
    def max_drawdown(self) -> float:
        self.df['peak'] = self.df['b.value'].cummax()
        self.df['drawdown'] = (self.df['peak'] - self.df['b.value']) / self.df['peak']
        return self.df['drawdown'].max()

    @property
    def duration(self) -> pd.Timedelta:
        return self.df.index[-1] - self.df.index[0]

    @property
    def entry_price(self) -> float:
        return self.df['close'].iloc[0]

    @property
    def current_price(self) -> float:
        return self.df['close'].iloc[-1] if self.is_open else None

    @property
    def exit_price(self) -> float:
        return self.df['close'].iloc[-1] if not self.is_open else None

    @property
    def is_open(self) -> bool:
        if self.is_long:
            return self.df['b.base'].iloc[-1] > 0
        elif self.is_short:
            return self.df['b.base'].iloc[-1] < 0

    def to_dict(self) -> dict:
        exit_time = self.df.index[-1].strftime("%Y-%m-%d %H:%M:%S") \
            if not self.is_open else None

        change_base_asset = self.df['b.base'].iloc[-1] - self.df['b.base'].iloc[-2]
        change_quote_asset = self.df['b.quote'].iloc[-1] - self.df['b.quote'].iloc[-2]

        return {
            "symbol": self.symbol,
            "position_type": self.position_type,
            "change_base_asset": change_base_asset,
            "change_quote_asset": change_quote_asset,
            "total_base_asset": self.df['b.base'].iloc[-1],
            "total_quote_asset": self.df['b.quote'].iloc[-1],
            "total_value": self.df['b.value'].iloc[-1],
            "pnl": self.pnl,
            "max_drawdown": self.max_drawdown,
            "duration": self.duration.total_seconds(),
            "entry_price": self.entry_price,
            "entry_time": self.df.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": self.current_price,
            "exit_price": self.exit_price,
            "exit_time": exit_time,
            "is_long": self.is_long,
            "is_short": self.is_short,
            "is_open": self.is_open,
        }

    def get_signal(self) -> dict:
        # self.df = self.df[0:-1]

        sell_size = float(self.df['sell_size'].iloc[-1])
        buy_size = float(self.df['buy_size'].iloc[-1])
        leverage = self.df['leverage'].iloc[-1]

        if buy_size > 0:
            required_action = 'BUY'
            change = 'increase' if self.is_long else 'decrease'
            change_percent = buy_size / self.df['b.base'].iloc[-2] * 100
        elif sell_size > 0:
            required_action = 'SELL'
            change = 'increase' if self.is_short else 'decrease'
            change_percent = sell_size / self.df['b.base'].iloc[-2] * 100
        else:
            required_action = None
            change = None
            change_percent = 0
            leverage = None

        entry_value = self.df['b.value'].iloc[0]
        exit_value = self.df['b.value'].iloc[-1]
        pnl_percentage = (exit_value - entry_value) / entry_value * 100

        return {
            "symbol": self.symbol,
            "position_type": self.position_type,
            "required_action": required_action,
            "change": change,
            "change_percent": change_percent,
            "leverage": leverage,
            "pnl_percentage": pnl_percentage,
            "max_drawdown": self.max_drawdown,
            "duration": self.duration.total_seconds(),
            "entry_price": self.entry_price,
            "entry_time": self.df.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": self.current_price,
            "exit_price": self.exit_price,
            "is_open": self.is_open,
        }


class PositionManager:
    def __init__(self):
        self.positions = []

    def __iter__(self):
        return iter(self.positions)

    def __len__(self):
        return len(self.positions)

    def add_position(self, position: Position):
        self.positions.append(position)

    def winning_percentage(self):
        winning_trades = sum(pos.pnl > 0 for pos in self.positions)
        total_trades = len(self.positions)
        return (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    def average_profit(self):
        profits = [pos.pnl for pos in self.positions if pos.pnl > 0]
        return np.mean(profits) if profits else 0

    def average_loss(self):
        losses = [pos.pnl for pos in self.positions if pos.pnl < 0]
        return np.mean(losses) if losses else 0

    @property
    def profit_factor(self):
        total_profit = sum(pos.pnl for pos in self.positions if pos.pnl > 0)
        total_loss = abs(sum(pos.pnl for pos in self.positions if pos.pnl < 0))
        return total_profit / total_loss if total_loss != 0 else float("inf")

    def total_pnl(self):
        return sum(pos.pnl for pos in self.positions)

    def average_duration(self):
        durations = [pos.duration for pos in self.positions]
        return pd.Timedelta(np.mean(durations))

    def get_current_position(self, symbol: str, as_dict: bool) -> Position | None:
        position = next(
            (
                pos for pos in reversed(self.positions)
                if (pos.symbol == symbol and pos.is_open)),
            None
            )

        if position is None:
            return None

        return position.to_dict() if as_dict else position


def extract_positions(backtest_df: pd.DataFrame, symbol: str) -> PositionManager:
    position_manager = PositionManager()

    # Group by position changes
    grouped = backtest_df.groupby(
        (backtest_df['position'] != backtest_df['position'].shift()).cumsum()
        )

    for _, group in grouped:
        if group['position'].iloc[0] != 0:  # Ignore periods with no position
            position_type = "long" if group['position'].iloc[0] == 1 else "short"
            position = Position(symbol, position_type, group)
            position_manager.add_position(position)

    return position_manager
