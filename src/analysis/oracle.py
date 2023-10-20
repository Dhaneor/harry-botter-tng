#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:58 2021

@author: dhaneor
"""
import sys
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging

from plotly.subplots import make_subplots
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import namedtuple
from numba import jit
import bottleneck as bn

from .util.find_positions import find_positions_with_dict
from .strategies.exit_order_strategies import sl_strategy_factory
from .strategies.exit_order_strategies import IExitOrderStrategy, StopLossDefinition
from .analysts import *

logger = logging.getLogger("main.oracle")
logger.setLevel(logging.ERROR)


# =============================================================================
LEVERAGE_PRECISION = 2
cache_entry = namedtuple("CacheEntry", ["request", "result", "expiry"])


# =============================================================================
@jit(nopython=True, cache=True)
def atr_nb(open_, high, low, period=14):
    """Caclulates ATR time series/array (Numba accelerated).

    :param open_: time series of open prices
    :type open_: np.nd_array
    :param high: time series of high prices
    :type high: np.nd_array
    :param low: time series of low prices
    :type low: np.nd_array
    :param period: ATR lookback period, defaults to 14
    :type period: int, optional
    :return: time series of ATR values
    :rtype: np.nd_array
    """
    atr = np.zeros(len(open_))
    for i in range(period, len(open_)):
        tr = max(high[i] - low[i], abs(high[i] - open_[i]), abs(low[i] - open_[i]))
        atr[i] = (atr[i - 1] * (period - 1) + tr) / period
    return atr


def calculate_atr(open_, high_, low_, atr_period=14):
    return (
        pd.Series(
            np.maximum.reduce(
                [high_ - low_, np.abs(low_ - open_), np.abs(high_ - open_)]
            )
        )
        .rolling(window=atr_period)
        .mean()
    )


@jit(nopython=True, cache=True)
def np_ffill(arr: np.ndarray):
    out = arr.copy()

    for row_idx in range(out.shape[0]):
        if np.isnan(out[row_idx]):
            out[row_idx] = out[row_idx - 1]
    return out


# ==============================================================================
@dataclass
class OracleSignal:
    symbol: str
    weight: float
    strategy: str
    data: Optional[pd.DataFrame] = None
    signal: Optional[int] = None
    target_position: Optional[str] = None
    target_leverage: Optional[float] = None
    stop_loss: Optional[Tuple[float, float]] = None
    take_profit: Optional[Tuple[float, float]] = None


@dataclass(unsafe_hash=True)
class OracleRequest:
    symbol: str = field(hash=True)
    interval: str = field(hash=True)
    strategy: str = field(hash=True)
    data: pd.DataFrame = field(compare=False, hash=False)


class OracleCache:
    def __init__(self, ttl: int = 5):
        self.cache: List[cache_entry] = []
        self.ttl: int = ttl
        self.max_size: int = 5000

    def get_result(self, request: OracleRequest) -> Union[pd.DataFrame, None]:
        for item in reversed(self.cache):
            try:
                if item.request == request and item.expiry > int(time.time()):
                    return item.result
            except:
                print(item.request.data.columns)
                print(request.data.columns)
                sys.exit()

        return None

    def cache_it(self, request: OracleRequest, result: pd.DataFrame):
        expiry = int(time.time()) + self.ttl
        item = cache_entry(request=request, result=result, expiry=expiry)
        self.cache.append(item)

        if len(self.cache) > self.max_size:
            del self.cache[0]


# =============================================================================
class Oracle:
    def __init__(self):
        self.name: str = "THE ORACLE"
        self.strategy_name: str = ""
        self.strategy: Optional[BaseStrategy] = None

        self.sl_strategy: Optional[IExitOrderStrategy] = None
        self.sl_params: Optional[dict] = None

        self.strategies: dict = {
            "Trend Is Friend": TrendIsFriend,
            "Eager Trend": EagerTrend,
            "Cautious Trend": CautiousTrend,
            "To The Contrary": ToTheContrary,
            "Secure Momentum": SecureMomentum,
            "Turn Around": TurnAround,
            "Breakout": Breakout,
            "Au Contraire": AuContraire,
            "Keltner Magic": KeltnerMagic,
            # --------------------------------------------
            # below are single indicator/signal
            # strategies
            "Pure Keltner": PureKeltner,
            "Pure Momentum": PureMomentum,
            "Pure Trend": PureTrend,
            "Pure Moving Average Cross": PureMovingAverageCross,
            "Pure RSI": PureRSI,
            "Pure Stochastic RSI": PureStochRSI,
            "Pure Stochastic": PureStochastic,
            "Pure Commodity Channel": PureCommodityChannel,
            "Pure Average Directional Index": PureADX,
            "Pure Choppiness Index": PureChoppinessIndex,
            "Pure Hammer": PureHammer,
            "Pure Breakout": PureBreakout,
            "Pure Disparity": PureDisparity,
            "Pure Connors RSI": PureConnorsRSI,
            "Pure Fibonacci Trend": PureFibonacciTrend,
            "Pure Trendy": PureTrendy,
            "Pure Big Trend": PureBigTrend,
        }

    # -------------------------------------------------------------------------
    def get_available_strategies(self) -> List[str]:
        """Returns all available strategies.

        :return: all possible strategies
        :rtype: List[str]
        """
        return [k for k, _ in self.strategies.items()]

    def set_strategy(self, strategy_name: str):
        """Sets the trading strategy.

        :param strategy_name: name of the trading strategy
        :type strategy_name: str
        :raises ValueError: if strategy_name is not a valid strategy
        """
        for k, v in self.strategies.items():
            if k == strategy_name:
                self.strategy_name = strategy_name
                self.strategy = v()
                logger.info(f"setting strategy to: {strategy_name}")
                return

        raise ValueError(f"{strategy_name} is not a valid strategy")

    def set_sl_strategy(
        self, sl_def: Optional[StopLossDefinition | dict] = None
    ) -> None:
        logger.debug(f"request for stop-loss strategy: {sl_def}")

        if isinstance(sl_def, dict):
            strategy = sl_def.get("strategy", None)
            params = {k: v for k, v in sl_def.items() if k != "strategy"}
            sl_def = StopLossDefinition(strategy, params)

        if sl_def == self.sl_params:
            return

        logger.debug(f"setting stop-loss strategy to: {sl_def}")

        if sl_def is None:
            self.sl_strategy = None
            self.sl_params = None
        else:
            self.sl_strategy = sl_strategy_factory(sl_def)

        self.sl_params = sl_def

    # -------------------------------------------------------------------------
    def speak(self, data: pd.DataFrame) -> dict:
        """Adds signals and positions to OHLCV dataframe.

        Parameters
        ----------
        data
            OHLCV datraframe

        Returns
        -------
        dcit
            dictionary with signals and (theoretical) positions
        """

        # extract OHLCV data as Numpy arrays to a dictionary. These will
        # be used in several Numba optimized functions later on and we
        # can save time by doing the conversion only once. Numba cannot
        # handle dataframes.
        ohlcv_dict = self._extract_numpy_arrays(data)

        # generate signals ...
        #
        # TODO: this should also use the dictionary, but
        # cannot be done yet, because the strategies expect a dataframe
        # and all strategies need to be refactored  ... which is not easy,
        # because some important functions (rolling fo instance) are not
        # implemented in Numpy :( it's way easier to do everything with
        # Pandas, it's just slow(er)

        if self.strategy:
            if self.strategy_name in ("Pure Keltner", "Pure RSI"):
                ohlcv_dict = self.strategy.get_signals(ohlcv_dict)
                self._extend_dictionary(data=ohlcv_dict)
            else:
                data = self.strategy.get_signals(data)
                for a in self.strategy.analysts:
                    ind_cols = a.plot_params.get("columns")
                    for col in ind_cols:
                        ohlcv_dict[col] = data[col]

                    sig_col = a.column_name
                    ohlcv_dict[sig_col] = data[sig_col]

                self._extend_dictionary(ohlcv_dict, data)

        else:
            raise ValueError("No strategy set. Unable to add signals.")

        # add stop-loss values for long and short trades if necessary
        if self.sl_strategy is not None:
            self._add_stop_loss(ohlcv_dict)

        # calculate positions from signals (and stop-loss values)
        find_positions_with_dict(ohlcv_dict)

        return ohlcv_dict

    # --------------------------------------------------------------------------
    def _extract_numpy_arrays(self, data: pd.DataFrame) -> dict:
        return {
            "interval_in_ms": int(np.min(data["open time"].diff())),
            "open time": data["open time"].to_numpy(),
            "human open time": data["human open time"].to_numpy(),
            "o": data.open.to_numpy(),
            "h": data.high.to_numpy(),
            "l": data.low.to_numpy(),
            "c": data.close.to_numpy(),
            "v": data.volume.to_numpy(),
        }

    def _extend_dictionary(
        self, data: Dict[str, np.ndarray], df: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        if df is not None:
            data["signal"] = df.signal.to_numpy()
            data["position"] = df.signal.ffill().shift().to_numpy()
        else:
            data["position"] = bn.push(data["signal"], axis=0)

        add_keys = [
            "leverage",
            "buy",
            "buy_size",
            "buy_at",
            "sell",
            "sell_size",
            "sell_at",
            "sl_current",
            "sl_trig",
            "tp_current",
            "tp_trig",
            "sl_long",
            "sl_short",
            "tp_long",
            "tp_short",
        ]

        length = len(data["o"])

        return {
            data.update(
                {key: np.full(length, np.nan, dtype=np.float64)}
            )
            for key in add_keys
            if key not in data.keys()
        }

    def _extend_ohlcv_dict(
        self, ohlcv_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        length = len(ohlcv_dict["o"])
        extension = {
            "position": bn.push(ohlcv_dict["signal"], axis=0),
            "leverage": np.zeros(length),
            "buy": np.zeros(length),
            "buy_size": np.zeros(length),
            "buy_at": np.zeros(length),
            "sell": np.zeros(length),
            "sell_size": np.zeros(length),
            "sell_at": np.zeros(length),
            "sl_current": np.zeros(length),
            "sl_trig": np.zeros(length),
            "tp_current": np.zeros(length),
            "tp_trig": np.zeros(length),
            "sl_long": np.zeros(length),
            "sl_short": np.zeros(length),
            "tp_long": np.zeros(length),
            "tp_short": np.zeros(length),
        }

        return ohlcv_dict.update(extension)

    def _add_stop_loss(self, ohlcv_dict: dict) -> None:
        if self.sl_strategy is not None:
            (
                ohlcv_dict["sl_long"],
                ohlcv_dict["sl_short"],
            ) = self.sl_strategy.get_trigger_prices_np(
                ohlcv_dict["o"], ohlcv_dict["h"], ohlcv_dict["l"], ohlcv_dict["c"]
            )

    def _cleanup(self, data: pd.DataFrame) -> pd.DataFrame:
        data.loc[data["position"].isna(), "leverage"] = np.nan

        drop_cols = [
            col
            for col in ("v", "event_id", "sl_long", "sl_short", "interval_in_ms")
            if col in data.columns
        ]

        data.drop(columns=drop_cols, inplace=True)

        data.rename(
            {"o": "open", "h": "high", "l": "low", "c": "close"}, axis=1, inplace=True
        )
        return data

    # -------------------------------------------------------------------------
    def draw_chart(self, df: pd.DataFrame):
        if self.strategy is not None:
            self.strategy.draw_chart(df=df)


class LiveOracle(Oracle):
    def __init__(self):
        super().__init__()
        self.cache = OracleCache(ttl=5)

    def speak(self, request: OracleRequest) -> pd.DataFrame:
        cached = self.cache.get_result(request)

        if cached is not None:
            return cached

        self.set_strategy(request.strategy)
        df = self.strategy.get_signals(df=request.data.copy(deep=True))

        # continuous signals ... this makes sure that a trade is entered
        # immediately after starting the bot. However, it also leads to
        # immediate re-entry after SL was triggered!
        df["signal"].replace(0, np.nan, inplace=True)
        df["signal"].ffill(inplace=True)

        if df is not None:
            self.cache.cache_it(request=request, result=df)
            return df
        else:
            raise Exception(f"unable to get signals from {self.strategy}")


# =============================================================================
#                                   STRATEGIES                                #
# =============================================================================
class BaseStrategy(ABC):
    def __init__(self):
        self.name: str = ""
        self.analysts: List[IAnalyst] = []

    # -------------------------------------------------------------------------
    @abstractmethod
    def get_signals(self, data: Union[dict, pd.DataFrame]) -> Union[dict, pd.DataFrame]:
        pass

    def draw_chart(self, df: pd.DataFrame, signal_colum: str = "signal"):
        """Draws a chart for the currently active strategy and its
        BUY/SELL signals.

        If the concrete strategy has populated the self.analysts list
        with instances derived from BaseSignalAnalyst then the method
        below will be called to include subpolots with the indicator
        values.

        :param df: a DataFrame that must include a signal column with
        : type df: pd.DataFrame
        :param signal_column: the name of the column containing
        BUY/SELL signals (signals format: -1/0/1/np.nan)
        :type signal_column: str
        """
        if "buy_at" not in df.columns:
            df.loc[(df[signal_colum] >= 1) & (df[signal_colum] <= 20), "buy_at"] = df[
                "open"
            ]
        if "sell_at" not in df.columns:
            df.loc[
                (df[signal_colum] <= -1) & (df[signal_colum] >= -20), "sell_at"
            ] = df["open"]

        if self.analysts:
            self._draw_extended_chart(df=df)
            return

        template = "presentation"  #'plotly_dark'
        buy_color = "chartreuse"
        sell_color = "red"

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["human open time"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    increasing_line_color="dimgrey",
                    decreasing_line_color="gray",
                )
            ]
        )

        if "kc.upper" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["human open time"],
                    y=df["kc.upper"],
                    visible=True,
                    name="KC upper",
                    line={"shape": "linear", "color": "azure"},
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df["human open time"],
                    y=df["kc.lower"],
                    visible=True,
                    name="KC lower",
                    line={"shape": "linear", "color": "azure"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["human open time"],
                    y=df["kc.mid"],
                    visible=True,
                    name="KC mid",
                    line={"shape": "linear", "color": "azure"},
                )
            )

        fig.update_traces(opacity=0.7, selector=dict(type="scatter"))
        fig.update_traces(line_width=0.5, selector=dict(type="scatter"))
        fig.update_traces(line_width=0.5, selector=dict(type="candlestick"))

        # ---------------------------------------------------------------------
        # add markers
        marker_size = 12
        marker_opacity = 1

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df["human open time"],
                y=df["buy_at"],
                marker=dict(
                    symbol="triangle-up",
                    color=buy_color,
                    size=marker_size,
                    opacity=marker_opacity,
                    line=dict(color=buy_color, width=1),
                ),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df["human open time"],
                y=df["sell_at"],
                marker=dict(
                    symbol="triangle-down",
                    color=sell_color,
                    size=marker_size,
                    opacity=marker_opacity,
                    line=dict(color=sell_color, width=1),
                ),
                showlegend=False,
            )
        )

        fig = self._fig_add_stop_loss(df, fig)

        bg_color = "antiquewhite"
        fig.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)

        fig.update_layout(template=template, yaxis_type="log", hovermode="x")
        # fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
        fig.show()

        return

    def _draw_extended_chart(self, df: pd.DataFrame):
        """Draws a chart for the currently active strategy and its
        BUY/SELL signals, including subplots for indicators used.
        """

        def zoom(layout, xrange):
            in_view = df.loc[fig.layout.xaxis.range[0] : fig.layout.xaxis.range[1]]
            fig.layout.yaxis.range = [in_view.High.min() - 10, in_view.High.max() + 10]

        template = "presentation"  # 'plotly_dark'
        buy_color = "chartreuse"
        sell_color = "red"

        analysts = [a for a in self.analysts if hasattr(a, "plot_params")]

        rows, row_heights, subplot_titles = 2, [4, 1], ["", "leverage"]
        [(rows := rows + 1) for a in analysts if hasattr(a, "plot_params")]
        [row_heights.append(2) for _ in range(rows - 2)]
        [subplot_titles.append(a.plot_params["label"]) for a in analysts]

        # prepare figure
        fig = make_subplots(
            rows=rows,
            cols=1,
            row_heights=row_heights,
            vertical_spacing=0.05,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
        )

        if "kc.upper" in df.columns:
            line_color = "thistle"  # 'palevioletred'

            fig.append_trace(
                go.Scatter(
                    x=df["human open time"],
                    y=df["kc.upper"],
                    visible=True,
                    name="KC upper",
                    line={"shape": "linear", "color": line_color},
                ),
                row=1,
                col=1,
            )

            fig.append_trace(
                go.Scatter(
                    x=df["human open time"],
                    y=df["kc.lower"],
                    visible=True,
                    name="KC lower",
                    fill="tonexty",
                    line={"shape": "linear", "color": line_color, "width": 2},
                ),
                row=1,
                col=1,
            )
            fig.append_trace(
                go.Scatter(
                    x=df["human open time"],
                    y=df["kc.mid"],
                    visible=True,
                    name="KC mid",
                    line={"shape": "linear", "color": line_color},
                ),
                row=1,
                col=1,
            )

        # draw candlesticks
        fig.append_trace(
            go.Candlestick(
                x=df["human open time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="dimgrey",
                decreasing_line_color="gray",
            ),
            row=1,
            col=1,
        )

        # add moving averages to candlestick plot
        for col in df.columns:
            if "ewm." in col and not "diff" in col:
                fig.append_trace(
                    go.Scatter(
                        mode="lines",
                        x=df["human open time"],
                        y=df[col],
                        name=col,
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

        # ---------------------------------------------------------------------
        # add markers
        marker_size = 8
        marker_opacity = 0.7

        fig.append_trace(
            go.Scatter(
                mode="markers",
                x=df["human open time"],
                y=df["buy_at"],
                marker=dict(
                    symbol="triangle-up",
                    color=buy_color,
                    size=marker_size,
                    opacity=marker_opacity,
                    line=dict(color=buy_color, width=1),
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.append_trace(
            go.Scatter(
                mode="markers",
                x=df["human open time"],
                y=df["sell_at"],
                marker=dict(
                    symbol="triangle-down",
                    color=sell_color,
                    size=marker_size,
                    opacity=marker_opacity,
                    line=dict(color=sell_color, width=1),
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig = self._fig_add_stop_loss(df, fig)

        # ----------------------------------------------------------------------
        # draw subplot for leverage
        df["leverage"].replace(np.nan, value=0, inplace=True)

        fig.append_trace(
            go.Scatter(
                mode="lines",
                x=df["human open time"],
                y=df["leverage"],
                name="leverage",
                showlegend=False,
                line={"shape": "hvh"},
            ),
            row=2,
            col=1,
        )

        # add indicator values for each analyst
        if rows > 2:
            for idx, a in enumerate(self.analysts):
                if hasattr(a, "plot_params"):
                    for col in a.plot_params["columns"]:
                        row = idx + 3
                        fig.append_trace(
                            go.Scatter(
                                mode="lines",
                                x=df["human open time"],
                                y=df[col],
                                name=a.plot_params["label"],
                                showlegend=False,
                            ),
                            row=row,
                            col=1,
                        )

                    # add channel to indicator plot
                    if a.plot_params["channel"]:
                        fig.append_trace(
                            go.Scatter(
                                mode="lines",
                                x=df["human open time"],
                                y=df[a.plot_params["channel"][0]],
                                name=a.plot_params["label"],
                            ),
                            row=row,
                            col=1,
                        )

                        fig.append_trace(
                            go.Scatter(
                                mode="lines",
                                x=df["human open time"],
                                y=df[a.plot_params["channel"][1]],
                                name=a.plot_params["label"],
                                fill="tonexty",
                            ),
                            row=row,
                            col=1,
                        )

                    # add horizontal lines to indicator plot
                    for line in a.plot_params["horizontal lines"]:
                        fig.add_hline(
                            y=line,
                            row=row,
                            col=1,
                            line_color="grey",
                            line_dash="dash",
                            opacity=0.5,
                        )

                    # add buy/sell signals to indicator plot
                    sig_col = a.plot_params["signal"]
                    df.loc[df[sig_col] > 0, "buy_signals"] = 1
                    df.loc[df[sig_col] < 0, "sell_signals"] = 1

                    for r in df.iterrows():
                        if r[1]["buy_signals"] > 0:
                            fig.add_vline(
                                x=df.at[r[0], "human open time"],
                                row=row,
                                col=1,
                                line_width=0.5,
                                line_color="darkgreen",
                                line_dash="solid",
                                opacity=1,
                            )

                        if r[1]["sell_signals"] > 0:
                            fig.add_vline(
                                x=df.at[r[0], "human open time"],
                                row=row,
                                col=1,
                                line_width=0.5,
                                line_color=sell_color,
                                line_dash="solid",
                                opacity=1,
                            )

        fig.update_layout(
            template=template, yaxis_type="log", hovermode="x", autosize=True
        )

        bg_color = "antiquewhite"
        fig.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)

        font_family = "Raleway"  # "Gravitas One"
        fig.update_layout(
            font_family=font_family,
            font_color="grey",
            font_size=10,
            title_font_family=font_family,
            title_font_color="red",
            legend_title_font_color="green",
            showlegend=False,
        )

        fig.update_xaxes(
            row=4,
            col=1,
            showgrid=False,
            zeroline=False,
            zerolinewidth=0,
            zerolinecolor="rgba(0,255,0,0.1)",
        )
        fig.update_traces(opacity=1, selector=dict(type="scatter"))
        fig.update_traces(line_width=0.7, selector=dict(type="scatter"))
        fig.update_traces(line_width=0.7, selector=dict(type="candlestick"))

        # fig.update_layout(xaxis=dict(rangeslider=dict(visible=True, )))
        fig.update_xaxes(row=1, col=1, rangeslider_visible=False)
        fig.update_xaxes(row=2, col=1, rangeslider_visible=False)
        fig.update_xaxes(rangeslider={"visible": True}, row=rows, col=1)
        fig.update_xaxes(row=rows, col=1, rangeslider_thickness=0.05)

        # fig.layout.font.family = "Old Standard TT"
        # fig.layout.on_change(zoom, 'xaxis.range')
        fig.show()

        return

    def _fig_add_stop_loss(self, df, fig):
        fig.add_trace(
            go.Scatter(
                x=df["human open time"],
                y=df["sl_current"],
                line={"shape": "hvh", "color": "rgba(255,77,77,0.5)", "width": 1},
                mode="lines",
                name="stop loss",
            )
        )

        # fig.add_trace(
        #     go.Scatter(
        #         x = df['human open time'],
        #         y = df["sl.short"],
        #         line = {
        #             "shape": 'hvh',
        #             'color' : 'rgba(25,102,255,0.5)',
        #             'width' : 1
        #         },
        #         mode = 'lines',
        #         name = 'SL short',
        #     )
        # )

        return fig


# =============================================================================
class TrendIsFriend(BaseStrategy):
    def __init__(self):
        self.name = "Trend Is Friend"

        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = RsiAnalyst()
        df = a.get_signal(df=df, lookback=14)

        a = StochRsiAnalyst()
        df = a.get_signal(df)

        a = TrendAnalyst()
        df = a.get_signal(df)

        a = MovingAverageCrossAnalyst()
        df = a.get_signal(df)

        a = MomentumAnalyst()
        df = a.get_signal(df)

        a = KeltnerChannelAnalyst()
        df = a.get_signal(df=df, multiplier=5)

        a = AverageDirectionalIndexAnalyst()
        df = a.get_signal(df=df, lookback=21, threshhold=18)

        df = self._combine_signals(df)

        return df

    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = (
            df["s.trnd"] + df["s.ma.x"] + df["s.rsi"] + df["s.stc.rsi"] + df["s.mom"]
        )

        df.loc[(df["signal"] > 0) & (df["s.adx"] == 0), "s.all"] = 0
        df.loc[(df["signal"] < 0) & (df["s.adx"] == 0), "s.all"] = 0

        df.loc[df["s.kc"] == -1, "s.all"] = -1
        df.loc[df["s.kc"].shift() == -1, "s.all"] = 1

        return df


# =============================================================================
class EagerTrend(BaseStrategy):

    """
    Don't change anymore!

    This strategy tries to capture uptrends and cannot be used for shorting.

    - seems to work best with trailing SL: ATR 1.75x
    - us only on USDT pairs, doesn't work on BTC pairs
    """

    def __init__(self):
        self.name = "Eager Trend"
        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # using the Trend Analyst just to determine market state
        a = TrendAnalyst()
        a.ma_type = "ewm"
        df = a.get_signal(df)

        for col in df.columns:
            if "slope" in col:
                df.drop(col, axis=1, inplace=True)

        # ---------------------------------------------------------------------
        # using the MA cross as entry signal
        a = MovingAverageCrossAnalyst()
        a.ma_type = "ewm"
        df = a.get_signal(df=df)

        # ---------------------------------------------------------------------
        # the Keltner Channel Analyst will find the exit points
        a = KeltnerChannelAnalyst()
        df = a.get_signal(df=df, multiplier=2)

        # ---------------------------------------------------------------------
        # the STOCH RSI will act as sanity check when buying
        a = StochRsiAnalyst()
        self.analysts.append(a)
        df = a.get_signal(
            df=df, method="crossover", period_rsi=14, period=5, k_period=3, d_period=3
        )

        df = self._combine_signals(df=df)

        return df

    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        df.loc[(df["ma.diff"] > 0), "s.all"] = 1

        df.loc[df["s.kc"] == -1, "s.all"] = -1

        df.loc[
            (df["stoch.rsi.k"] < df["stoch.rsi.d"]) & (df["s.trnd"] == 1), "s.all"
        ] = 0

        df.loc[(df["s.trnd"] < 0), "s.all"] = -1

        df.loc[(df["close"] <= df["ewm.63"]) & (df["signal"] == 1), "s.all"] = -1
        df.loc[(df["close"] >= df["ewm.63"]) & (df["signal"] == -1), "s.all"] = 1

        return df

    # -------------------------------------------------------------------------
    # def draw_chart(self, df):

    #     cols_subplot_1 = ['Human Open Time', 'close', 'sma.7', 'sma.21']
    #     cols_subplot_2 = ['RSI Close']
    #     cols_subplot_3 = ['slope.close', 'slope.sma.7', 'slope.sma.21']
    #     cols_signals = ['s.trnd', 's.ma.x', 's.rsi']

    #     oc = OracleChart(df)
    #     oc.draw_signal_chart(cols_subplot_1, cols_subplot_2, cols_subplot_3, cols_signals)

    #     return


# =============================================================================
class CautiousTrend(BaseStrategy):

    """
    This strategy tries to capture uptrends and cannot be used for shorting.
    It is more conservative than 'Eager Trend' and sell when the StochRSI
    crosses to the downside (coming from an oversold condition)
    """

    def __init__(self):
        self.name = "Cautious Trend"

        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # using the Trend Analyst just to determine market state
        a = TrendAnalyst()
        a.ma_type = "ewm"
        df = a.get_signal(df)

        # ---------------------------------------------------------------------
        # we need the Momentum Analyst to find entry points
        # a = MomentumAnalyst()
        # df = a.get_signal(df=df)

        # ---------------------------------------------------------------------
        # using the MA cross as entry signal
        a = MovingAverageCrossAnalyst()
        a.ma_type = "ewm"
        df = a.get_signal(df=df)

        # ---------------------------------------------------------------------
        # the Keltner Channel Analyst will find the exit points
        a = KeltnerChannelAnalyst()
        df = a.get_signal(df=df, multiplier=3.3)

        # ---------------------------------------------------------------------
        # the STOCH RSI will act as sanity check when buying
        a = StochRsiAnalyst()
        a.overbought, a.oversold = 80, 20
        df = a.get_signal(df=df, method="crossover")

        df = self._combine_signals(df=df)

        return df

    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        df.loc[(df["ma.diff"] > 0) & (df["ma.diff"].diff() >= 0), "s.all"] = 1

        df.loc[(df["stoch.rsi.diff"] <= 0), "s.all"] = 0
        df.loc[(df["stoch.rsi.diff"] > 20), "s.all"] = 0

        df.loc[((df["s.trnd"] < 0)) & (df["signal"] == 1), "s.all"] = -1

        df.loc[df["s.kc"] == -1, "s.all"] = -1

        df.loc[(df["s.stc.rsi"] < 0), "s.all"] = 0

        df.loc[(df["close"] <= df["ewm.63"]) & (df["signal"] == 1), "s.all"] = 0
        df.loc[(df["close"] >= df["ewm.63"]) & (df["signal"] == -1), "s.all"] = 0

        return df


# =============================================================================
class ToTheContrary(BaseStrategy):
    def __init__(self):
        self.name = "To The Contrary"

        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        overbought, oversold = 70, 30

        # ---------------------------------------------------------------------
        a = CommodityChannelAnalyst()
        a.overbought, a.oversold = 125, -125

        df = a.get_signal(df=df, period=20)

        # ---------------------------------------------------------------------
        a = StochasticAnalyst()
        a.overbought, a.oversold = overbought, oversold

        df = a.get_signal(df)

        # ---------------------------------------------------------------------
        a = StochRsiAnalyst()
        a.overbought, a.oversold = overbought, oversold

        df = a.get_signal(df=df, method="crossover")

        # ---------------------------------------------------------------------
        a = RsiAnalyst()
        a.overbought, a.oversold = overbought, oversold

        df = a.get_signal(df=df, lookback=14)

        # ---------------------------------------------------------------------
        a = KeltnerChannelAnalyst()
        a.kc_lookback = 14
        a.atr_lookback = 14
        df = a.get_signal(df=df, multiplier=3)

        # ---------------------------------------------------------------------
        a = MovingAverageCrossAnalyst()
        df = a.get_signal(df=df)

        a = TrendAnalyst()
        df = a.get_signal(df=df)

        a = AverageDirectionalIndexAnalyst()
        df = a.get_signal(df=df)

        # ---------------------------------------------------------------------
        df = self._combine_signals(df=df)

        return df

    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = df["s.cci"] + df["s.stc"] + df["s.stc.rsi"] + df["s.kc"]
        df["s.exit"] = df["s.cci"] + df["s.stc"] + df["s.stc.rsi"]

        # ---------------------------------------------------------------------
        min_signals = 3
        df.loc[abs(df["signal"]) < min_signals, "s.all"] = 0

        df.loc[(df["s.trnd"] > 0) & (df["s.exit"] < 2), "s.all"] = 1
        df.loc[(df["s.trnd"] < 0) & (df["signal"] >= 1), "s.all"] = 0

        df.loc[df["s.ma.x"] == -1, "s.all"] = -1
        # df.loc[(df['s.exit'] <= -2) & (df['close'] > df['kc.upper']), 's.all'] = -1

        df.loc[(df["s.exit"] <= -2) & (df["signal"] >= 1), "s.all"] = 0

        return df

    def _add_cool_off(self, df: pd.DataFrame, length: int = 5) -> pd.DataFrame:
        # don't enter any new trades during a cool-off period of 5 intervals
        # after selling
        df["cool.off"] = False
        df.loc[df["signal"].rolling(length).min() < 0, "cool.off"] = True
        df.loc[(df["signal"] == 1) & df["cool.off"] == True, "s.all"] = 0

        # ... the same for stop-loss triggered
        df["sl.trig"] = False
        df.loc[df["Low"] < (df["close"].shift() * df["sl.l.pct"]), "s.all"] = True

        df.loc[
            df["signal"].rolling(length).max().astype(bool) == True, "cool.off"
        ] = True

        cols = [
            "Human Open Time",
            "open",
            "high",
            "low",
            "close",
            "s.all",
            "sl.trig",
            "cool.off",
        ]

        # print(df.loc[:, cols].tail(50))
        # sys.exit()
        return df


# =============================================================================
class SecureMomentum(BaseStrategy):
    """ "Secure Momentum uses the Momentum Indicator for entry and exit
    signals. Positions are secured by a rather complicated stop-loss
    strategy that sets a tight stop-loss on entry so that losing
    positions are closed rather fast.

    The tight stop-loss allows using leverage (5-10x).
    """

    def __init__(self):
        self.name = "Secure Momentum"

        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---------------------------------------------------------------------
        a = KeltnerChannelAnalyst()
        a.kc_lookback = 14
        a.atr_lookback = 14
        df = a.get_signal(df=df, multiplier=2.5)

        # ---------------------------------------------------------------------
        a = MomentumAnalyst()
        a.lookback = 9
        df = a.get_signal(df=df)

        # ---------------------------------------------------------------------
        a = MovingAverageAnalyst()
        df = a.get_signal(df=df)

        # ---------------------------------------------------------------------
        df["signal"] = df["s.mom"]
        df.loc[(df["signal"] == 1) & (df["high"] > df["kc.upper"]), "s.all"] = 0
        df.loc[
            (df["signal"] == 1) & (df["high"].shift() > df["kc.upper"].shift()), "s.all"
        ] = 0

        df.loc[(df["signal"] == 1) & (df["close"] < df["ewm.63"]), "s.all"] = 0
        df.loc[(df["signal"] == -1) & (df["close"] > df["ewm.63"]), "s.all"] = 0

        return df


# =============================================================================
class TurnAround(BaseStrategy):
    """ "Turn Around" looks for candlestick patterns that may indicate a
    change in trend and seeks confirmation by checking the StochRSI and
    the CCI"""

    def __init__(self):
        self.name = "Turn Around"
        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---------------------------------------------------------------------
        a = WickAndBodyAnalyst()
        df = a.get_signal(df=df, confirmation=True)

        # ---------------------------------------------------------------------
        a = StochRsiAnalyst()
        a.overbought, a.oversold = 70, 30
        df = a.get_signal(df=df)

        # ---------------------------------------------------------------------
        # a = RsiAnalyst()
        # a.overbought, a.oversold = 70, 30
        # df = a.get_signal(df=df)

        # ---------------------------------------------------------------------
        conditions = [
            (df["s.wab"] + df["s.stc.rsi"] == 2),
            (df["s.wab"] + df["s.stc.rsi"] == 2),
        ]
        choices = [1, -1]
        df["signal"] = np.select(conditions, choices, default=0)
        return df


class Breakout(BaseStrategy):
    """Classic Breakout strategy"""

    def __init__(self):
        self.name = "Turn Around"
        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---------------------------------------------------------------------
        a = BreakoutAnalyst()
        df = a.get_signal(df=df, lookback=30)

        a = MovingAverageAnalyst()
        df = a.get_signal(df=df, period=63)

        # ---------------------------------------------------------------------
        conditions = [
            (df["s.bo"] > 0) & (df["s.ma"] > 0),
            (df["s.bo"] > 0) & (df["s.ma"] < 0),
            (df["s.bo"] < 0) & (df["s.ma"] < 0),
            (df["s.bo"] < 0) & (df["s.ma"] > 0),
        ]
        choices = [1, -1, -1, 1]
        df["signal"] = np.select(conditions, choices, default=0)

        return df


class AuContraire(BaseStrategy):
    """Mean reversion strategy"""

    def __init__(self):
        self.name = "Au Contraire"
        super().__init__()

    def get_signals(
        self, df: pd.DataFrame, oversold: int = 20, overbought: int = 80
    ) -> pd.DataFrame:
        # ---------------------------------------------------------------------
        a = StochasticAnalyst()
        self.analysts.append(a)

        df = a.get_signal(df=df, oversold=oversold, overbought=overbought)
        df["signal"] = df["s.stc"]

        a = AtrMomentumAnalyst()
        self.analysts.append(a)
        df = a.get_signal(df=df, lookback=21)

        # df.loc[~(df['s.stc'] == 0) & (df['s.atr.mom'] == -1), 's.all'] = 0

        return self.filter(df)

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        a = ChoppinessIndexAnalyst()
        self.analysts.append(a)
        df = a.get_signal(df=df)

        a = MovingAverageAnalyst()
        self.analysts.append(a)
        df = a.get_signal(df=df, period=5)

        # df.loc[~(df['signal'] == 0) & (df['s.ci'] == -1), 's.all'] = 0

        df.loc[~(df["signal"] == df["s.ma"]), "s.all"] = 0
        return df


class KeltnerMagic(BaseStrategy):
    """Strategy based on Keltner Channel and Stochastic RSI as filter.

    long entry:     Close below lower band + Stoch RSI %d-line is rising
    short entry:    Close above upper band + Stoch RSI %d-line is falling

    long SL:        maximum: 2x ATR/ entry price/ lower band/ mid band
    short SL:       minimum:  2x ATR/ entry price/ upper band/ mid band

    """

    def __init__(self):
        super().__init__()
        self.name = "Pure Keltner"

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = KeltnerChannelAnalyst()
        df = a.get_signal(df=df, multiplier=2, mode=0)

        a = StochRsiAnalyst()
        a.get_signal(df=df, period=14, method="extremes")
        self.analysts.append(a)

        df["signal"] = np.nan
        df.loc[(df["s.kc"] == 1) & (df["s.stc.rsi"] == 1), "s.all"] = 1
        df.loc[(df["s.kc"] == -1) & (df["s.stc.rsi"] == -1), "s.all"] = -1
        df.loc[(df["s.kc"] == 0), "s.all"] = 0

        return df


# =============================================================================
#                    Single Indicator Strategies for Testing                  #
# =============================================================================
class PureKeltner(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Pure Keltner"

    def get_signals(self, data: dict) -> Dict[str, Any]:
        a = KeltnerChannelAnalyst()
        data = a.get_signal(data_as_dict=data, multiplier=2)
        data["signal"] = data["s.kc"]

        return data


# =============================================================================
class PureMomentum(BaseStrategy):
    def __init__(self):
        self.name = "Pure Momentum"

        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = MomentumAnalyst()
        a.lookback = 9
        df = a.get_signal(df=df)
        df["signal"] = df["s.mom"]

        a = KeltnerChannelAnalyst()
        df = a.get_signal(df=df, multiplier=3)

        return df


# =============================================================================
class PureTrend(BaseStrategy):
    def __init__(self):
        self.name = "Pure Trend"

        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = TrendAnalyst()
        a.ma_short = 16
        a.ma_long = 64
        df = a.get_signal(df=df)
        df["signal"] = df["s.trnd"]

        a = KeltnerChannelAnalyst()
        df = a.get_signal(df=df)

        return df


# =============================================================================
class PureFibonacciTrend(BaseStrategy):
    def __init__(self):
        self.name = "Pure Fibonacci Trend"

        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = FibonacciTrendAnalyst()
        self.analysts = [a]

        df = a.get_signal(df=df)
        df["signal"] = df[a.column_name]

        return df


# =============================================================================
class PureMovingAverageCross(BaseStrategy):
    def __init__(self):
        self.name = "Pure Moving Average Cross"

        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = MovingAverageCrossAnalyst()
        # a.ma_short = 9
        # a.ma_long = 36
        df["signal"] = a.get_signal(df.close.to_numpy())

        return df


# =============================================================================
class PureRSI(BaseStrategy):
    def __init__(self):
        self.name = "Pure RSI"
        super().__init__()
        self.lookback: int = 21

    def get_signals(self, data: dict) -> dict:
        a = RsiAnalyst()
        if not self.analysts:
            self.analysts.append(a)

        data = a.get_signal(data=data, mode=0, lookback=self.lookback)
        data["signal"] = data["s.rsi"]

        return data


# =============================================================================
class PureStochRSI(BaseStrategy):
    def __init__(self):
        self.name = "Pure Stochastic RSI"
        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = StochRsiAnalyst()
        df = a.get_signal(df=df)
        self.analysts.append(a)

        # reverse signal
        df["s.stc.rsi"] = df["s.stc.rsi"] * -1

        df["signal"] = df["s.stc.rsi"]

        return df


# =============================================================================
class PureStochastic(BaseStrategy):
    def __init__(self):
        self.name = "Pure Stochastic"
        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = StochasticAnalyst()
        a.period = 21
        a.oversold = 2
        a.overbought = 99
        self.analysts.append(a)

        df = a.get_signal(df=df)
        df["signal"] = df["s.stc"]

        return df


# =============================================================================
class PureCommodityChannel(BaseStrategy):
    def __init__(self):
        self.name = "Pure Commodity Channel"
        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = CommodityChannelAnalyst()
        a.period = 30
        a.overbought = 150
        a.oversold = -150
        self.analysts.append(a)

        df = a.get_signal(df=df)
        df["signal"] = df["s.cci"] * -1

        return df


# =============================================================================
class PureADX(BaseStrategy):
    def __init__(self):
        self.name = "Pure Average Directional Index"
        super().__init__()

    def get_signals(
        self, df: pd.DataFrame, lookback: int = 14, threshhold: int = 25
    ) -> pd.DataFrame:
        a = AverageDirectionalIndexAnalyst()
        self.analysts.append(a)

        df = a.get_signal(df=df, lookback=14, threshhold=threshhold)
        df["signal"] = df["s.adx"]

        return df


class PureChoppinessIndex(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Pure Choppiness Index"

    def get_signals(
        self, df: pd.DataFrame, lookback: int = 14, atr_lookback: int = 14
    ) -> pd.DataFrame:
        a = ChoppinessIndexAnalyst()
        df = a.get_signal(df=df, lookback=lookback, atr_lookback=atr_lookback)
        df["signal"] = df["s.ci"]

        return df


class PureConnorsRSI(BaseStrategy):
    def __init__(self):
        self.name = "Pure Connors RSI"
        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = ConnorsRsiAnalyst()
        a.overbought, a.oversold = 70, 30
        df = a.get_signal(df=df, rsi_lookback=7, streak_lookback=7, smoothing=5)
        df["signal"] = df["s.c_rsi"]

        return df


class PureDisparity(BaseStrategy):
    def __init__(self):
        self.name = "Pure Disparity"
        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = DisparityAnalyst()
        df = a.get_signal(df=df)
        df["signal"] = df["s.disp"]

        return df


class PureTrendy(BaseStrategy):
    def __init__(self):
        self.name = "Pure Trendy"
        super().__init__()

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = TrendyAnalyst()
        self.analysts.append(a)
        df = a.get_signal(df=df, threshhold=2, lookback=14, smoothing=3)
        df["signal"] = df[a.column_name]

        return df


class PureBigTrend(BaseStrategy):
    """The Big Trend strategy calculates volatility bands for a
    given lookback period and enters a  position when the close
    price moves outside of this band.
    """

    name: str = "Pure Big Trend"

    def __init__(self):
        super().__init__()

    def get_signals(
        self,
        df: pd.DataFrame,
        period: int = 20,
        factor: float = 0.00007,
        fast_exit=False,
    ) -> pd.DataFrame:
        """Add signal to the dataframe for 'Big Trend' strategy.

        The standard lookback period is 20 and seems to work best. The
        factor determines the sensitivity by adjusting the width of
        the volatility bands. It's very sensitive and results can change
        significantly by adjusting it.

        NOTE:   The sources that I found all mentioned a factor of about 0.001.
                However, this makes the bands so wide that there are no signals
                at all. This seems strange ...

        :param df: OHLCV dataframe
        :type df: pd.DataFrame
        :param period: lookback period for Big Trend, defaults to 20
        :type period: int, optional
        :param factor: determines sensitivity/width of volatility bands,
        defaults to 0.00007
        :type factor: float, optional
        :param fast_exit: exit fast or slow, defaults to False
        :type fast_exit: bool, optional
        :return: OHLCV dataframe with signal
        :rtype: pd.DataFrame
        """
        a = BigTrendAnalyst()
        self.analysts.append(a)

        df = a.get_signal(df=df, period=period, factor=factor, fast_exit=fast_exit)

        df["signal"] = df[a.column_name]

        return df


# =============================================================================
#                               CANDLESTICK PATTERNS                          #
# =============================================================================


class PureHammer(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Pure Hammer"

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = WickAndBodyAnalyst()
        df = a.get_signal(df=df, confirmation=True)
        df["signal"] = df["s.wab"]

        return df


class PureBreakout(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Pure Breakout"

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        a = BreakoutAnalyst()
        df = a.get_signal(df=df, lookback=30)
        df["signal"] = df["s.bo"]

        return df
