#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 11 22:18:20 2024

@author dhaneor
"""
from datetime import datetime, UTC, timedelta
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("main.position")

Change = Literal["open", "close", "increase", "decrease"] | None
Action = Literal["buy", "sell"] | None


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

        leverage = (
            f"target leverage: {self.target_leverage:.2f}x"
            if not self.change == "close"
            else ""
        )

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
        is_long = self.row.position in (1, "LONG")
        self.position_type = "LONG" if is_long else "SHORT"

        # determine change type (open / close / increase / decrease)
        if (self.row.action == "BUY" and self.position_type == "LONG") or (
            self.row.action == "SELL" and self.position_type == "SHORT"
        ):
            self.change = "increase"
        elif (self.row.action == "SELL" and self.position_type == "LONG") or (
            self.row.action == "BUY" and self.position_type == "SHORT"
        ):
            self.change = "decrease"
        else:
            self.change = None

        if self.period == 0:
            self.change = "open"

        if self.row.change_percent == -100:
            self.change = "close"

        # determine the required action (buy or sell)
        self.action = self.row.action if self.row.action != "None" else None
        self.change_percent = self.row.change_percent

        self.target_leverage = self.row.leverage


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
        return self.df["b.value"].iloc[-1] - self.df["b.value"].iloc[0]

    @property
    def max_drawdown(self) -> float:
        self.df["peak"] = self.df["b.value"].cummax()
        self.df["drawdown"] = (self.df["peak"] - self.df["b.value"]) / self.df["peak"]
        return self.df["drawdown"].max()

    @property
    def entry_time(self) -> pd.Timestamp:
        # Convert Pandas Timestamp to Python datetime
        return self.df.index[0].to_pydatetime().replace(tzinfo=UTC)

    @property
    def entry_time_utc(self):
        try:
            return self.entry_time.strftime("%B %d, %H:%M")
        except Exception as e:
            logger.exception(e)
            logger.error("\n%s" % self.df)
            raise

    @property
    def exit_time(self) -> datetime:
        # Convert Pandas Timestamp to Python datetime
        return self.df.index[-1].to_pydatetime().replace(tzinfo=UTC)

    @property
    def duration(self) -> pd.Timedelta:
        if self.is_open:
            now = datetime.now(tz=UTC)
            return (now - self.entry_time).total_seconds()

        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def entry_price(self) -> float:
        return self.df["close"].iloc[0]

    @property
    def current_price(self) -> float:
        return self.df["close"].iloc[-1] if self.is_open else None

    @property
    def exit_price(self) -> float:
        return self.df["close"].iloc[-1] if not self.is_open else None

    @property
    def is_new(self) -> bool:
        return self.df.index[0] == self.df.index[-1]

    @property
    def is_open(self) -> bool:
        if self.is_long:
            return self.df["b.base"].iloc[-1] > 0
        elif self.is_short:
            return self.df["b.base"].iloc[-1] < 0
        else:
            return False

    def display_df(self) -> pd.DataFrame:
        incl_cols = [
            "close",
            "position",
            "buy",
            "buy_size",
            "sell",
            "sell_size",
            "leverage",
            "change_percent",
            "b.base",
            "b.quote",
            "b.value",
            "period",
            "action",
            "pos_size_change",
            "last_action_leverage",
            "change_percent_1",
        ]

        return (
            self.df.copy()
            .replace([0, "None", np.nan], [".", "", ""])
            .drop(columns=[col for col in self.df.columns if col not in incl_cols])
            .round(5)
        )

    def to_dict(self) -> dict:
        exit_time = (
            self.df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            if not self.is_open
            else None
        )

        change_base_asset = self.df["b.base"].iloc[-1] - self.df["b.base"].iloc[-2]
        change_quote_asset = self.df["b.quote"].iloc[-1] - self.df["b.quote"].iloc[-2]

        return {
            "symbol": self.symbol,
            "position_type": self.position_type,
            "change_base_asset": change_base_asset,
            "change_quote_asset": change_quote_asset,
            "total_base_asset": self.df["b.base"].iloc[-1],
            "total_quote_asset": self.df["b.quote"].iloc[-1],
            "total_value": self.df["b.value"].iloc[-1],
            "pnl": self.pnl,
            "max_drawdown": self.max_drawdown,
            "duration": self.duration,
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
        """
        Generate a signal dictionary containing current position
        information and trading signals.

        This method compiles various attributes of the current position,
        including the most recent trade, into a dictionary format. This
        information can be used for decision-making in trading strategies
        or for position monitoring.

        Returns:
            dict: A dictionary containing the following keys:
                - symbol (str): The trading symbol of the position.
                - position_type (str): The type of position
                (e.g., 'LONG' or 'SHORT').
                - required_action (str): The action required based
                on the last trade (e.g., 'BUY' or 'SELL').
                - change (str): The type of change in the position
                (e.g., 'open', 'close', 'increase', 'decrease').
                - change_percent (float): The absolute percentage change
                in the position size.
                - leverage (float): The target leverage for the position.
                - pnl_percentage (float): The percentage of profit or loss
                for the position.
                - max_drawdown (float): The maximum drawdown experienced
                by the position.
                - duration (pd.Timedelta): The duration of the position.
                - entry_price (float): The entry price of the position.
                - entry_time (str): The entry time of the position in UTC.
                - current_price (float): The current price of the asset.
                - exit_price (float): The exit price of the position
                (if closed).
                - is_open (bool): Whether the position is currently open.
                - is_new (bool): Whether the position is newly opened.
        """
        lt = self.trades[-1]  # get the last trade

        return {
            "symbol": self.symbol,
            "position_type": self.position_type,
            "required_action": lt.action,
            "change": lt.change,
            "change_percent": abs(lt.change_percent),
            "leverage": lt.target_leverage,
            "pnl_percentage": lt.row.pnl_percent,
            "max_drawdown": self.max_drawdown,
            "duration": self.duration,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time_utc,
            "current_price": self.current_price,
            "exit_price": self.exit_price,
            "is_open": self.is_open,
            "is_new": self.is_new,
        }

    def _extract_trade_actions(self, df: pd.DataFrame) -> list[TradeAction]:
        """
        Extract trade actions from a DataFrame and calculate various metrics.

        This method processes the input DataFrame to extract trade actions and
        calculate additional metrics such as period, action type, position size
        changes, PNL, and drawdown.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing trade data.
                           Expected to have columns such as 'buy_size', 'sell_size',
                           'b.base', and 'b.value'.

        Returns:
        list[TradeAction]: A list of TradeAction objects, each representing a trade
                           action derived from the processed DataFrame.

        Note:
        This method modifies the input DataFrame by adding new columns.
        """
        # add some columns that we will need later on in each single
        # TradeAction object for signal generation

        # the period for each row
        df["period"] = np.arange(len(df))

        # which action (buy or sell) occurred in each row
        df["action"] = np.where(
            self.df["buy_size"] > 0,
            "BUY",
            np.where(df["sell_size"] > 0, "SELL", "None"),
        )

        # how much did the position size change
        df["pos_size_change"] = df["b.base"].diff()
        df["change_percent"] = (df.pos_size_change / df["b.base"].shift(1)) * 100
        df.change_percent = df.change_percent.replace(np.nan, 100)

        # what is the PNL for each row/period
        df["pnl_percent"] = (df["b.value"].iloc[-1] / df["b.value"].iloc[0] - 1) * 100
        df["max_drawdown"] = (df["b.value"].div(df["b.value"].cummax()) - 1) * 100

        # return a list of all trade actions with a change in position size
        return [
            ta for ta in (TradeAction(row, self.symbol) for row in df.itertuples())
        ]


class Positions:
    def __init__(self, df: pd.DataFrame, symbol: str):
        self.symbol = symbol
        self.positions = self._extract_positions(df, symbol)

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

    def current(self, as_dict: bool = False) -> Position | None:
        position = next((p for p in reversed(self.positions) if p.is_open), None)

        return position.to_dict() if (as_dict and position) else position

    def _extract_positions(self, backtest_df: pd.DataFrame, symbol: str) -> list[Position]:
        positions = []

        # Group by position changes
        grouped = backtest_df.groupby(
            (backtest_df["position"] != backtest_df["position"].shift()).cumsum()
        )

        for _, group in grouped:
            if group["position"].iloc[0] != 0:  # Ignore periods with no position
                position_type = "long" if group["position"].iloc[0] == 1 else "short"
                positions.append(Position(symbol, position_type, group))

        return positions
