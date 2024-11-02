#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:58 2021

@author: dhaneor
"""
import sys  # noqa: F401
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

from matplotlib.axes import Axes
from typing import Tuple, Optional, Union

logger = logging.getLogger("main.minerva")
logger.setLevel("DEBUG")

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)


# =============================================================================
class Minerva:
    def __init__(self):
        self.name: str = "MINERVA"
        self.df: pd.DataFrame
        self.height_top_chart: int = 6
        self.no_of_subplots: int
        self.title: str = ""
        self.color_scheme: str

    # -------------------------------------------------------------------------
    # methods for plotting the different subplots
    def _ohlcv(self, with_market_state: bool = False):
        """Plot all the OHLCV candles - color coded for market state"""

        def _get_ax_size(ax):
            bbox = ax.get_window_extent().transformed(
                self.fig.dpi_scale_trans.inverted()
            )  # type:ignore
            width, height = bbox.width, bbox.height
            width *= self.fig.dpi
            height *= self.fig.dpi
            return width, height

        def _get_linewidth() -> tuple:
            df = self.df
            values = len(self.df)
            precision = 5

            open_time = df.index.astype(int)
            timedelta = int((open_time[1] - open_time[0]) / 1_000_000_000)

            _, ax_width = _get_ax_size(ax)
            corr_factor = 86400 / timedelta
            value_factor = values / 1000

            width = round(
                (
                    (ax_width * 0.5) / (values) * value_factor
                ) / corr_factor,
                precision
            )

            width2 = round(width / 3, precision)
            return width, width2

        # ......................................................................
        try:
            ax = self.axes[0]
        except Exception:
            ax = self.axes

        ax.set_title(self.title, y=1.0, pad=-14)

        width, width2 = _get_linewidth()

        bull, alpha_bull = self.bull[0], self.bull[1]
        bear, alpha_bear = self.bear[0], self.bear[1]
        flat, alpha_flat = self.flat[0], self.flat[1]

        if with_market_state:
            # plot candles with color depending on market regime

            if "s.state" not in self.df.columns:
                self._add_trend_signal(df=self.df)

            for x_, row in self.df.iterrows():
                _open, _high = row["open"], row["high"]
                _low, _close = row["low"], row["close"]
                state = row["s.state"]

                if state == "bull":
                    if _close > _open:
                        ax.bar(
                            x_,
                            height=_close - _open,
                            width=width,
                            bottom=_open,
                            color=bull,
                            alpha=alpha_bull,
                        )
                        ax.bar(
                            x_,
                            height=_high - _close,
                            width=width2,
                            bottom=_close,
                            color=bull,
                            alpha=alpha_bull,
                        )
                        ax.bar(
                            x_,
                            height=_open - _low,
                            width=width2,
                            bottom=_low,
                            color=bull,
                            alpha=alpha_bull,
                        )
                    else:
                        ax.bar(
                            x=x_,
                            height=_open - _close,
                            width=width,
                            bottom=_close,
                            color=bull,
                            alpha=alpha_bull,
                        )
                        ax.bar(
                            x=x_,
                            height=_high - _open,
                            width=width2,
                            bottom=_open,
                            color=bull,
                            alpha=alpha_bull,
                        )
                        ax.bar(
                            x=x_,
                            height=_close - _low,
                            width=width2,
                            bottom=_low,
                            color=bull,
                            alpha=alpha_bull,
                        )

                elif state == "bear":
                    if _close > _open:
                        ax.bar(
                            x_,
                            height=_close - _open,
                            width=width,
                            bottom=_open,
                            color=bear,
                            alpha=alpha_bear,
                        )
                        ax.bar(
                            x_,
                            height=_high - _close,
                            width=width2,
                            bottom=_close,
                            color=bear,
                            alpha=alpha_bear,
                        )
                        ax.bar(
                            x_,
                            height=_open - _low,
                            width=width2,
                            bottom=_low,
                            color=bear,
                            alpha=alpha_bear,
                        )
                    else:
                        ax.bar(
                            x=x_,
                            height=_open - _close,
                            width=width,
                            bottom=_close,
                            color=bear,
                            alpha=alpha_bear,
                        )
                        ax.bar(
                            x=x_,
                            height=_high - _open,
                            width=width2,
                            bottom=_open,
                            color=bear,
                            alpha=alpha_bear,
                        )
                        ax.bar(
                            x=x_,
                            height=_close - _low,
                            width=width2,
                            bottom=_low,
                            color=bear,
                            alpha=alpha_bear,
                        )

                elif state == "flat":
                    if _close > _open:
                        ax.bar(
                            x_,
                            height=_close - _open,
                            width=width,
                            bottom=_open,
                            color=flat,
                            alpha=alpha_flat,
                        )
                        ax.bar(
                            x_,
                            height=_high - _close,
                            width=width2,
                            bottom=_close,
                            color=flat,
                            alpha=alpha_flat,
                        )
                        ax.bar(
                            x_,
                            height=_open - _low,
                            width=width2,
                            bottom=_low,
                            color=flat,
                            alpha=alpha_flat,
                        )

                    else:
                        ax.bar(
                            x=x_,
                            height=_open - _close,
                            width=width,
                            bottom=_close,
                            color=flat,
                            alpha=alpha_flat,
                        )
                        ax.bar(
                            x=x_,
                            height=_high - _open,
                            width=width2,
                            bottom=_open,
                            color=flat,
                            alpha=alpha_flat,
                        )
                        ax.bar(
                            x=x_,
                            height=_close - _low,
                            width=width2,
                            bottom=_low,
                            color=flat,
                            alpha=alpha_flat,
                        )

        else:
            # plot candles normally (up=green, down=red)
            for x_, row in self.df.iterrows():
                _open, _high = row["open"], row["high"]
                _low, _close = row["low"], row["close"]

                if _close > _open:
                    ax.bar(
                        x_,
                        height=_high - _close,
                        width=width2,
                        bottom=_close,
                        color=bull,
                        alpha=alpha_bull,
                    )
                    ax.bar(
                        x_,
                        height=_open - _low,
                        width=width2,
                        bottom=_low,
                        color=bull,
                        alpha=alpha_bull,
                    )
                    ax.bar(
                        x_,
                        height=_close - _open,
                        width=width,
                        bottom=_open,
                        color=bull,
                        alpha=alpha_bull,
                    )
                else:
                    ax.bar(
                        x=x_,
                        height=_high - _open,
                        width=width2,
                        bottom=_open,
                        color=bear,
                        alpha=alpha_bear,
                    )
                    ax.bar(
                        x=x_,
                        height=_close - _low,
                        width=width2,
                        bottom=_low,
                        color=bear,
                        alpha=alpha_bear,
                    )
                    ax.bar(
                        x=x_,
                        height=_open - _close,
                        width=width,
                        bottom=_close,
                        color=bear,
                        alpha=alpha_bear,
                    )

    def _buys_and_sells(self):
        """Plot markers for buy/sell signals"""
        try:
            ax = self.axes[0]
        except Exception:
            ax = self.axes

        markersize = 3

        x = self.df.index.to_list()

        self.df = self.df.replace({'buy_at': {0: np.nan}})
        self.df = self.df.replace({'sell_at': {0: np.nan}})

        ax.scatter(
            x,
            y=self.df["buy_at"],
            alpha=self.buy[1],
            marker="^",
            color=self.buy[0],
            s=markersize,
        )

        ax.scatter(
            x=self.df.index,
            y=self.df["sell_at"],
            alpha=self.sell[1],
            marker="v",
            color=self.sell[0],
            s=markersize,
        )

    def _positions(self):
        # plot vertical lines for candles with active position
        lw = 50 / (len(self.df) / 5)

        if "position" in self.df.columns:
            for x_ in list(self.df.index.values):
                try:
                    if self.df.at[x_, "position"] == "LONG":
                        self.axes[0].axvline(
                            x=x_,
                            color=self.buy[0],
                            alpha=self.position[1],
                            linewidth=lw,
                            linestyle="-",
                            zorder=-5,
                        )

                    if self.df.at[x_, "position"] == "SHORT":
                        self.axes[0].axvline(
                            x=x_,
                            color=self.sell[0],
                            alpha=self.position[1],
                            linewidth=lw,
                            linestyle="-",
                            zorder=-5,
                        )
                except ValueError:
                    print(x_)

    def _channel(self, subplots: Optional[dict] = None):
        if subplots is None:
            logger.warning("no subplots provided")
            return

        ax = self.axes[0] if isinstance(subplots, np.ndarray) else self.axes

        for name, params in subplots.items():
            logger.debug(params)

            if not params.get("channel"):
                continue

            try:
                upper = params.get("channel")[0]
                lower = params.get("channel")[1]
                middle = params.get("columns")[0]
            except KeyError as e:
                upper, lower, middle = None, None, None
                logger.exception(e)
            finally:
                if not upper and lower and middle:
                    logger.debug(
                        "unable to determine upper and/or lower bound for channel"
                    )
                    return

            ax.plot(
                self.df[upper], color=self.channel[0], linestyle="dotted", linewidth=0.1
            )

            ax.plot(
                self.df[lower], color=self.channel[0], linestyle="dotted", linewidth=0.1
            )

            ax.plot(
                self.df[middle],
                color=self.channel[0],
                linestyle="dotted",
                linewidth=0.1,
            )

            if params.get("fill"):
                logger.debug(f'filling channel between "{upper}" and "{lower}"')

                ax.fill_between(
                    x=self.df.index,
                    y1=self.df[lower],
                    y2=self.df[upper],
                    color=self.channel_bg[0],
                    edgecolor=self.channel[0],
                    alpha=self.channel_bg[1],
                    linewidth=0.1,
                    zorder=-5,
                )

    def _moving_averages(self):
        try:
            ax = self.axes[0]
        except Exception:
            ax = self.axes

        colors = [self.line1[0], self.line2[0], self.line3[0], self.line4[0], "black"]
        alpha = [
            self.line1[1],
            self.line2[1],
            self.line3[1],
            self.line4[1],
            self.line4[1],
        ]

        columns = [
            col
            for col in self.df.columns
            if col.split(".")[0] == "sma" or col.split(".")[0] == "ewm"
            and "diff" not in col
        ]

        for idx, col in enumerate(columns):
            lw = 0.5

            try:
                color = colors[idx]
                alpha = alpha[idx] * (1 / (idx + 1))
            except Exception:
                color = "grey"
                alpha = 0.5

            label = col.split(".")[0].upper() + " " + col.split(".")[1]
            ax.plot(self.df[col], color=color, linewidth=lw, alpha=alpha, label=label)

    def _stop_loss(self):
        ax = self.axes[0]

        if "sl_current" in self.df.columns:
            ax.plot(
                self.df["sl_current"],
                color=self.line1[0],
                linewidth=0.1,
                alpha=self.line1[1] * 2,
                drawstyle="steps-mid",
            )

    def _leverage(self):
        self.df["leverage"] = self.df["leverage"].replace("", np.NaN)
        self.df = self.df.replace({'buy_at': {0: np.nan}})

        ax = self.axes[-4]

        ax.plot(
            self.df["leverage"],
            color=self.line_colors[3],
            alpha=self.line_alphas[3],
            linewidth=0.5,
            label="leverage",
        )

    def _position_size(self):
        ax = self.axes[-3]
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_minor_locator(ticker.MaxNLocator(8))

        _, linewidth = self._get_linewidth(ax)
        linewidth = round(linewidth * 10, 5)

        if "b.base" not in self.df.columns:
            return

        ax.plot(
            self.df["b.base"],
            color=self.line_colors[1],
            alpha=self.line_alphas[1],
            linewidth=0.3,
            drawstyle="steps-pre",
            label="Position Size",
        )

        # ax.fill_between(
        #     x=self.df.index, y1=self.df['b.base'], y2=0,
        #     color=self.channel_bg[0], edgecolor=self.channel[0],
        #     alpha=self.channel_bg[1], linewidth=0.1, zorder=-5,
        #     label=None
        # )

        # only for ccomparison purposes, if alternative position size
        # was calculated and added to DataFrame
        if "p.size.alt" in self.df.columns:
            ax.plot(
                self.df["p.size.alt"],
                color=self.line_colors[1],
                linewidth=0.2,
                alpha=self.line_alphas[1],
                drawstyle="steps-mid",
                label="Alt Position Size",
            )

    def _drawdown(self):
        ax = self.axes[-2]
        self.df["cptl.drawdown"] = self.df["cptl.drawdown"].replace("", np.NaN)
        self.df["cptl.drawdown"] = self.df["cptl.drawdown"] * 100 * -1
        self.df["b.drawdown"] = self.df["b.drawdown"].replace("", np.NaN)
        self.df["b.drawdown"] = self.df["b.drawdown"] * 100 * -1
        self.df["hodl.drawdown"] = self.df["hodl.drawdown"].replace("", np.NaN)
        self.df["hodl.drawdown"] = self.df["hodl.drawdown"] * 100 * -1

        ax.fill_between(
            x=self.df.index,
            y1=self.df["hodl.drawdown"],
            y2=0,  # self.df["cptl.b"],
            color=self.line_colors[0],
            edgecolor=self.line_colors[0],
            alpha=self.channel_bg[1] / 2,
            linewidth=0.1,
            zorder=-5,
        )
        ax.plot(
            self.df["hodl.drawdown"],
            color=self.line_colors[0],
            linewidth=0.3,
            alpha=self.line_alphas[0],
            linestyle="dotted",
            label="HODL Drawdown",
        )

        ax.fill_between(
            x=self.df.index,
            y1=self.df["b.drawdown"],
            y2=0,  # self.df["cptl.b"],
            color=self.channel_bg[0],
            edgecolor=self.channel[0],
            alpha=self.channel_bg[1] * 2,
            linewidth=0.1,
            zorder=-4,
        )

        ax.plot(
            self.df["b.drawdown"],
            color=self.line_colors[3],
            linewidth=0.3,
            alpha=self.line_alphas[3],
            label="Strategy Drawdown",
        )

        # ax.fill_between(
        #     x=self.df.index,
        #     y1=self.df["cptl.drawdown"].astype(np.float64),
        #     y2=0,
        #     color=self.channel_bg[0],
        #     edgecolor=self.channel[0],
        #     alpha=self.channel_bg[1],
        #     linewidth=0.1,
        #     zorder=-5,
        # )

    def portfolio_value(self):
        ax = self.axes[-1]

        ax.fill_between(
            x=self.df.index,
            y1=self.df["b.value"],
            y2=self.df["cptl.b"],
            color=self.channel_bg[0],
            edgecolor=self.channel[0],
            alpha=self.channel_bg[1],
            linewidth=0.1,
            zorder=-5,
        )

        ax.plot(
            self.df["b.value"],
            color=self.line_colors[3],
            alpha=self.line_alphas[3],
            linewidth=0.4,
            label="strategy",
        )

        if "hodl.value" in self.df.columns:
            hodl_color, hodl_alpha = self.line_colors[0], self.line_alphas[0]

            ax.plot(
                self.df["hodl.value"],
                color=hodl_color,
                alpha=hodl_alpha,
                linestyle="dotted",
                linewidth=0.5,
                zorder=-6,
                label="HODL",
            )

            self.axes[-1].tick_params(
                axis="y", labelcolor=hodl_color, labelsize=self.fontsize
            )

    # -------------------------------------------------------------------------
    # general settings for colors and formatting

    def prepare(self):
        print(f"{self.no_of_subplots} subplots")

        self._set_colors(color_scheme=self.color_scheme)

        figsize = (12, 5.2)

        if self.no_of_subplots > 1:
            ratios = [self.height_top_chart]
            for _ in range(self.no_of_subplots - 1):
                ratios.append(1)
            ratios[-1] += 3

            self.fig, self.axes = plt.subplots(
                self.no_of_subplots,
                1,
                figsize=figsize,
                sharex=True,
                sharey=False,
                gridspec_kw={"width_ratios": [1], "height_ratios": ratios},
            )
        else:
            self.fig, self.axes = plt.subplots(1, 1, figsize=figsize)

        self.fig.set_dpi(300)
        self.fig.patch.set_facecolor(self.canvas)

    def _set_colors(self, color_scheme: str):
        # import the styles as defined in the (mandatory) file mpl_schemes.py
        # which must reside in the same directory
        from plotting.mpl_styles import schemes as s

        allowed = [scheme for scheme in s.keys()]

        # set color scheme to 'day' (=default) if the given name for the color
        # scheme is not defined in mpl_styles
        color_scheme = color_scheme if color_scheme in allowed else "day"

        # set all colors from the imported color scheme
        self.canvas = s[color_scheme]["canvas"]
        self.background = s[color_scheme]["background"]
        self.tick = s[color_scheme]["tick"]

        self.bull = s[color_scheme]["bull"]
        self.bear = s[color_scheme]["bear"]
        self.flat = s[color_scheme]["flat"]

        self.buy = s[color_scheme]["buy"]
        self.sell = s[color_scheme]["sell"]
        self.position = s[color_scheme]["position"]

        self.line1 = s[color_scheme]["line1"]
        self.line2 = s[color_scheme]["line2"]
        self.line3 = s[color_scheme]["line3"]
        self.line4 = s[color_scheme]["line4"]

        self.channel = s[color_scheme]["channel"]
        self.channel_bg = s[color_scheme]["channel_bg"]

        self.grid = s[color_scheme]["grid"]
        self.hline = s[color_scheme]["hline"]

        # make a list of colors/alpha values for lines for easy looping
        self.line_colors = [self.line1[0], self.line2[0], self.line3[0], self.line4[0]]
        self.line_alphas = [self.line1[1], self.line2[1], self.line3[1], self.line4[1]]

    def _set_format_parameters(self):
        # different format parameters
        axes_list = self.axes if self.no_of_subplots > 1 else [self.axes]

        for ax_ in axes_list:
            ax_.set_facecolor(self.background)
            ax_.tick_params(axis="x", labelsize=self.fontsize, colors=self.tick[0])
            ax_.tick_params(axis="y", labelsize=self.fontsize, colors=self.tick[0])

            ax_.grid(
                which="both",
                linewidth=0.075,
                linestyle="dotted",
                color=self.grid[0],
                alpha=self.grid[1],
            )

            ax_.margins(tight=True)

            # Check if there are any labeled artists
            if ax_.get_legend_handles_labels()[0]:
                ax_.legend(
                    fancybox=True,
                    framealpha=0.5,
                    shadow=False,
                    borderpad=1,
                    labelcolor=self.grid[0],
                    facecolor=self.canvas,
                    fontsize=self.fontsize,
                    edgecolor=self.canvas,
                )

            for brdr in ["left", "right", "top", "bottom"]:
                ax_.spines[brdr].set_color(self.position[0])
                ax_.spines[brdr].set_linewidth(0.3)

    # -------------------------------------------------------------------------
    # helper methods for handling/preparing the dataframe

    def _columns_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make sure that the columns we need are numeric values ...
        """
        columns = [
            "open",
            "high",
            "low",
            "close",
            "buy.price",
            "sell.price",
            "b.base",
            "b.quote",
            "SMA7",
            "SMA21",
            "mom",
            "mom.sma.3",
            "mom.sma.7",
            "rsi.close.14",
            "sl.current",
            "b.value",
            "stoch.close.k",
            "stoch.close.d",
            "stoch.rsi.k",
            "stoch.rsi.d",
            "stoch.rsi.diff",
        ]

        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)  # df[col].apply(pd.to_numeric)

        return df

    def _add_trend_signal(
        self, df: pd.DataFrame, trend_signal: str = "s.adx"
    ) -> pd.DataFrame:
        """Checks if we have a trend signal (bull/bear/flat) in the
        DataFrame and adds it if necessary

        Possible trend signals (= their columns) are:

        's.adx' :   Average Directional Index (ADX) indicator
        's.trnd' :  my own indicator for market state (this is the default)
        """
        if trend_signal == "s.adx":
            if "s.adx" not in df.columns:
                from analysis.analysts import AverageDirectionalIndexAnalyst

                a = AverageDirectionalIndexAnalyst()
                df = a.get_signal(df=df, lookback=30)

        if trend_signal == "s.ci":
            if "s.ci" not in df.columns:
                from analysis.analysts import ChoppinessIndexAnalyst

                a = ChoppinessIndexAnalyst()
                df = a.get_signal(df=df)

        elif trend_signal == "s.trnd":
            if "s.trnd" not in df.columns:
                from analysis.analysts import TrendAnalyst

                columns_before = df.columns

                a = TrendAnalyst()
                df = a.get_signal(df=df)

                for col in df.columns:
                    if col not in columns_before and col != "s.trnd":
                        del df[col]

        df["s.state"] = "flat"
        df.loc[df[trend_signal] == 1, "s.state"] = "bull"
        df.loc[df[trend_signal] == -1, "s.state"] = "bear"

        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # df.dropna(inplace=True)
        # df = self._add_trend_signal(df=df)

        # if this dow not work, it probably means that 'human open time'
        # is already a datetime column and set as index
        df = self._columns_to_numeric(df=df)

        # generate a signal columns with random values if there is no
        # signal column
        # if not 's.all' in df.columns:
        #     print(f'[WARNING] no buy/sell signals ... generating random signals')
        #     rng = np.random.default_rng()
        #     df['s.all'] = rng.choice(3, len(df), p=[0.05, 0.9, 0.05]) - 1

        # fill in buy/sell prices from the signal column if necessary
        if "s.all" not in df.columns:
            df["s.all"] = np.nan

        if "buy.price" not in df.columns:
            df["buy.price"] = np.nan
            df.loc[df["s.all"].shift() > 0, "buy.price"] = df["open"]

        if "sell.price" not in df.columns:
            df["sell.price"] = np.nan
            df.loc[df["s.all"].shift() < 0, "sell.price"] = df["open"]

        return df

    def _get_ax_size(self, ax: Axes) -> Tuple[int, int]:
        bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= self.fig.dpi
        height *= self.fig.dpi
        return width, height

    def _get_linewidth(self, ax: Axes) -> Tuple[float, float]:
        df = self.df
        values = len(self.df)
        precision = 5

        open_time = df.index.astype(int)
        timedelta = int((open_time[1] - open_time[0]) / 1_000_000_000)
        print(f"{timedelta=}")

        _, ax_width = self._get_ax_size(ax)
        corr_factor = 86400 / timedelta
        value_factor = values / 10000

        width = round(((ax_width) / (values) * value_factor) / corr_factor, precision)

        width2 = round(width / 2, precision)

        return width, width2


# =============================================================================
class BasicChart(Minerva):
    def __init__(
        self,
        df: pd.DataFrame,
        title: str = "Unnamed Chart",
        color_scheme: str = "night",
    ):
        Minerva.__init__(self)

        self.df = self._prepare_dataframe(df)

        self.title = title
        self.counter = 2
        self.no_of_subplots = 1
        self.color_scheme = color_scheme
        self.fontsize = 4
        self.x = "human open time"

    def draw(self):
        self.prepare()
        self._ohlcv()

        # plot the whole thing
        plt.xlabel("date", color=self.tick[0], fontsize=self.fontsize)
        plt.tight_layout()
        self._set_format_parameters()
        plt.show()


# =============================================================================
class BacktestChart(Minerva):
    def __init__(
        self,
        df: pd.DataFrame,
        title: str = "Unnamed Chart",
        color_scheme: str = "night",
    ):
        Minerva.__init__(self)

        self.df = self._prepare_dataframe(df)

        self.title = title
        self.counter = 1
        self.no_of_subplots = 5
        self.color_scheme = color_scheme
        self.fontsize = 3
        self.x = "human open time"

    def draw(self):
        self.prepare()

        # ---------------------------------------------------------------------
        self._channel()
        self._positions()
        self._stop_loss()
        self._moving_averages()
        self._ohlcv(with_market_state=False)
        self._buys_and_sells()

        self._position_size()
        self._leverage()
        self._drawdown()
        self.portfolio_value()

        # ---------------------------------------------------------------------
        self._set_format_parameters()

        # plot the whole thing
        plt.xlabel("date", color=self.tick[0], fontsize=self.fontsize)
        plt.tight_layout()
        plt.show()


# =============================================================================
class AnalystChart(Minerva):
    def __init__(
        self,
        df: pd.DataFrame,
        subplots: Optional[dict] = None,
        color_scheme: str = "night",
        title: str = "",
        with_market_state: bool = False,
    ):
        Minerva.__init__(self)

        self.df: pd.DataFrame = self._prepare_dataframe(df)
        self.color_scheme: str = color_scheme

        self.subplots: Union[dict, None] = subplots

        if subplots:
            self.draw_on_main = {
                k: v for k, v in subplots.items() if v.get("main", False)
            }
            self.sub_plots = {
                k: v for k, v in subplots.items() if not v.get("main", False)
            }

        self.no_of_subplots: int = len(self.subplots) + 1
        self.counter: int = 1

        self.title: str = title
        self.fontsize: int = 4
        self.x: str = "human open time"

        self.with_market_state: bool = with_market_state

    def draw(self):
        # ---------------------------------------------------------------------
        # prepare subplotes
        self.prepare()
        self._channel(self.draw_on_main)
        self._moving_averages()
        self._ohlcv(with_market_state=self.with_market_state)
        self._buys_and_sells()

        if self.subplots and isinstance(self.subplots, dict):
            for indicator_name in self.subplots.keys():
                self._indicator(indicator_name)

        # ---------------------------------------------------------------------
        # different format parameters
        axes_list = self.axes if self.no_of_subplots > 1 else [self.axes]

        for ax_ in axes_list:
            ax_.set_facecolor(self.background)
            ax_.tick_params(axis="x", labelsize=self.fontsize, colors=self.tick[0])
            ax_.tick_params(axis="y", labelsize=self.fontsize, colors=self.tick[0])

            ax_.grid(
                which="both",
                linewidth=0.075,
                linestyle="dotted",
                color=self.grid[0],
                alpha=self.grid[1],
            )

            ax_.margins(tight=True)

            ax_.legend(
                fancybox=True,
                framealpha=0.5,
                shadow=False,
                borderpad=1,
                labelcolor=self.grid[0],
                facecolor=self.canvas,
                fontsize=self.fontsize,
                edgecolor=self.canvas,
            )

            for brdr in ["left", "right", "top", "bottom"]:
                ax_.spines[brdr].set_color(self.position[0])
                ax_.spines[brdr].set_linewidth(0.3)

        # ---------------------------------------------------------------------
        # plot the whole thing
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    def _indicator(self, indicator_name: str):
        params = self.subplots[indicator_name]
        print("subplot:", params.get("label"))
        ax = self.axes[self.counter]

        for idx, col in enumerate(params["columns"]):
            ax.plot(
                self.df[col],
                color=self.line_colors[idx],
                linewidth=0.5,
                alpha=self.line_alphas[idx],
                label=indicator_name,
            )

        for idx, line in enumerate(params["horizontal lines"]):
            ax.axhline(y=line, color=self.hline[0], linestyle="dotted", linewidth=0.5)

        if params["fill"]:
            if len(params["horizontal lines"]) >= 2:
                ax.fill_between(
                    x=self.df.index,
                    y1=params["horizontal lines"][0],
                    y2=params["horizontal lines"][1],
                    color=self.channel_bg[0],
                    alpha=self.channel_bg[1],
                )
        try:
            if params["channel"]:
                col_upper = params["channel"][0]
                col_lower = params["channel"][1]

                ax.fill_between(
                    x=self.df.index,
                    y1=self.df[col_upper],
                    y2=self.df[col_lower],
                    color=self.channel_bg[0],
                    alpha=self.channel_bg[1],
                )
        except Exception:
            pass

        if params.get("signal") is not None:
            sig_col = params["signal"]

            for idx in self.df.index:
                x_ = idx

                if self.df.loc[idx, sig_col] == 1:
                    ax.axvline(
                        x_,
                        color=self.buy[0],
                        linewidth=0.2,
                        alpha=self.channel_bg[1],
                    )

                if self.df.loc[idx, sig_col] == -1:
                    ax.axvline(
                        x_, color=self.sell[0], linewidth=0.2, alpha=self.channel_bg[1]
                    )

        self.counter += 1
