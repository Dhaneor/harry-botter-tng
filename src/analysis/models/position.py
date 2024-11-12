#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 10 22:18:20 2024

@author dhaneor
"""
import pandas as pd
import numpy as np
import time
from dataclasses import dataclass
from typing import Literal

Change = Literal['open', 'close', 'increase', 'decrease'] | None
Action = Literal['buy', 'sell'] | None


@dataclass
class TradeAction:
    row: pd.Series
    symbol: str
    timestamp: pd.Timestamp | None = None
    time_utc: str | None = None
    period: int | None = None
    position_type: str | None = None
    change: Change = None
    action: Action = None
    change_percent: float = 0.0
    target_leverage: float = 0.0

    def __repr__(self):
        change_pct = (
            f"by {abs(self.change_percent):.2f}%"
            if 0.00 < abs(self.change_percent) < 100
            else "\t"
        )

        leverage = f"target leverage: {self.target_leverage:.2f}x" \
            if not self.change == 'close' else ""

        return (
            f"[{self.period:03d}][{self.time_utc}] {self.change} "
            f"\t{self.position_type} {change_pct} \t{leverage}"
        )

    def __post_init__(self):
        # set timestamp and period
        self.timestamp = self.row[1]
        self.time_utc = self.row[2]
        self.period = self.row.period

        # determine position type (long or short)
        is_long = self.row.position in (1, 'LONG')
        self.position_type = 'LONG' if is_long else 'SHORT'

        # determine change type (open / close / increase / decrease)
        if (self.row.action == 'BUY' and self.position_type == 'LONG') \
                or (self.row.action == 'SELL' and self.position_type == 'SHORT'):
            self.change = 'increase'
        elif (self.row.action == 'SELL' and self.position_type == 'LONG') \
                or (self.row.action == 'BUY' and self.position_type == 'SHORT'):
            self.change = 'decrease'
        else:
            self.change = ''

        if self.period == 0:
            self.change = 'open'

        if self.row.change_percent == -100:
            self.change = 'close'

        # determine the required action (buy or sell)
        self.action = self.row.action if self.row.action != 'None' else None
        self.change_percent = self.row.change_percent

        self.target_leverage = self.row.leverage

    def as_signal(self) -> dict:
        return {
            "symbol": self.symbol,
            "position_type": self.position_type,
            "required_action": self.action,
            "change": self.change,
            "change_percent": self.change_percent,
            "leverage": self.target_leverage,
            "pnl_percentage": self.row.pnl_percent,
            "max_drawdown": self.row.max_drawdown,
            "duration": self.duration,
            "entry_price": None,  # self.entry_price,
            "entry_time": None,  # self.df.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": self.row.close,
            "exit_price": self.close if self.row['b.base'] == 0 else None,
            "is_open": True if self.row['b.base'] == 0 else False,
            "is_new": True if self.period == 0 else False,
        }


class Position:
    def __init__(self, symbol: str, position_type: str, df: pd.DataFrame):
        self.symbol = symbol
        self.position_type = position_type.upper()
        self.df: pd.DataFrame = df
        self.trades: list[TradeAction] = self._extract_trade_actions(df)

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
        print(self.df["open time"].iloc[0])

        if self.is_open:
            return time.time() - self.df["open time"].iloc[0]
        return (self.df.index[-1] - self.df.index[0]).total_seconds()

    @property
    def entry_time(self) -> pd.Timestamp:
        return self.df.index[0]

    @property
    def open_time_utc(self):
        return self.df.index[0].strftime("%Y-%m-%d %H:%M:%S")

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
    def exit_time(self) -> pd.Timestamp:
        return self.df.index[-1] if not self.is_open else None

    @property
    def is_new(self) -> bool:
        return self.df.index[0] == self.df.index[-1]

    @property
    def is_open(self) -> bool:
        if self.is_long:
            return self.df['b.base'].iloc[-1] > 0
        elif self.is_short:
            return self.df['b.base'].iloc[-1] < 0

    def display_df(self) -> pd.DataFrame:
        incl_cols = [
            'close', 'position', 'buy', 'buy_size', 'sell', 'sell_size',
            'leverage', 'change_percent',
            'b.base', 'b.quote', 'b.value',
            'period', 'action', 'pos_size_change', 'last_action_leverage',
            'change_percent_1'
        ]

        return (
            self.df
            .copy()
            .replace([0, 'None', np.nan], ['.', '', ''])
            .drop(columns=[col for col in self.df.columns if col not in incl_cols])
            .round(5)
        )

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
            "is_new": self.is_new,
        }

    def get_signal(self) -> dict:
        lt = self.trades[-1]  # get the last trade

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
            leverage = 0

        # entry_value = self.df['b.value'].iloc[0]
        # exit_value = self.df['b.value'].iloc[-1]
        # pnl_percentage = (exit_value - entry_value) / entry_value * 100

        return {
            "symbol": self.symbol,
            "position_type": self.position_type,
            "required_action": lt.action,
            "change": lt.change,
            "change_percent": change_percent,
            "leverage": leverage,
            "pnl_percentage": lt.row.pnl_percent,
            "max_drawdown": self.max_drawdown,
            "duration": self.duration,
            "entry_price": self.entry_price,
            "entry_time": self.df.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": self.current_price,
            "exit_price": self.exit_price,
            "is_open": self.is_open,
            "is_new": self.is_new,
        }

    def _extract_trade_actions(self, df: pd.DataFrame) -> list[TradeAction]:
        # add some columns that we will need later on in each single
        # TradeAction object for signal generation

        # the period for each row
        df["period"] = np.arange(len(df))

        # which action (buy or sell) occurred in each row
        df['action'] = np.where(
            self.df['buy_size'] > 0, 'BUY',
            np.where(df['sell_size'] > 0, 'SELL', 'None')
            )

        # how much did the position size change
        df['pos_size_change'] = df['b.base'].diff()
        df['change_percent'] = (df.pos_size_change / df['b.base'].shift(1)) * 100
        df.change_percent = df.change_percent.replace(np.nan, 100)

        # what is the PNL for each row/period
        df['pnl_percent'] = (df['b.value'].iloc[-1] / df['b.value'].iloc[0] - 1) * 100
        df['max_drawdown'] = (df['b.value'].div(df['b.value'].cummax()) - 1) * 100

        print(self.display_df())

        # return a list of all trade actions with a change in position size
        return [
            ta for ta in
            (TradeAction(row, self.symbol) for row in df.itertuples())
            # if ta.change
            ]


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
            (p for p in reversed(self.positions) if (p.symbol == symbol and p.is_open)),
            None
        )

        return position.to_dict() if as_dict and position else position


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
