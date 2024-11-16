#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 11 22:30:20 2024

@author dhaneor
"""
import logging
import os
import requests

from functools import wraps
from typing import Callable, Dict, Any, List
from functools import partial

logger = logging.getLogger('main.telegram_signal')
logger.setLevel(logging.INFO)

bot_token = os.getenv("BOT_TOKEN")

UP_ARROW = "\U0001F4C8"  # "\U0001F53C"  # 🔼
DOWN_ARROW = "\U0001F4C9"  # "\U0001F53D"  # 🔽


# --------------------------------- Helper functions ---------------------------------
def add_position_overview(func):
    def fv(value, width=6):
        """Format the value with the specified width,
        accounting for floats, integers, and None."""
        if isinstance(value, float):
            return f"`{value:>{width}.2f}`"  # .rstrip('0').rstrip('.')
        elif isinstance(value, int):
            return f"`{value:>{width},d}`"
        elif value is None:
            return " " * width
        else:
            return f"`{str(value):<{width}}`"

    def entry_time(position: Dict[str, Any]) -> str:
        return f"*Entry:*  \t{fv(position.get('entry_time'))}\n"

    def duration(position: Dict[str, Any]) -> str:
        return f"*Duration:*  \t{fv(_format_duration(position.get('duration')))}\n"

    def entry_price(position: Dict[str, Any]) -> str:
        return f"\n*Entry Price:*  \t{fv(position.get('entry_price'))}\n"

    def current_price(position: Dict[str, Any]) -> str:
        return f"*Current Price:* {fv(position.get('current_price'))}\n\n"

    def pnl(position: Dict[str, Any]) -> str:
        return f"*Profit:* {fv(position.get('pnl_percentage'))}`%`\n"

    def max_drawdown(position: Dict[str, Any]) -> str:
        return f"*Drawdown:* {fv((position.get('max_drawdown') * 100))}`%`\n\n"

    def exit_price(position: Dict[str, Any]) -> str:
        return f"*Exit Price:* {fv(position.get('exit_price'))}\n\n"

    def leverage(position: Dict[str, Any]) -> str:
        return f"*Leverage:* {fv(position.get('leverage'))}\n"

    pipelines = {
        "open": [
            entry_time, entry_price, leverage
            ],
        "close": [
            entry_time, duration, entry_price, exit_price, pnl, max_drawdown
            ],
        "increase": [
            entry_time, duration, entry_price, current_price,
            pnl, max_drawdown, leverage
            ],
        "decrease": [
            entry_time, duration, entry_price, current_price,
            pnl, max_drawdown, leverage
            ],
        "default": [
            entry_time, duration, entry_price, current_price, pnl,  max_drawdown
            ]
    }

    def build_overview(position: Dict[str, Any], pipeline: List[Callable]) -> str:
        return (
            f"\n\n*{position.get('symbol')} - "
            f"{position.get('position_type')} position*\n"
            + "".join(func(position) for func in pipeline)
        )

    @wraps(func)
    def wrapper(position: Dict[str, Any]) -> str:
        signal_message = func(position)
        change = position.get('change', 'default')
        pipeline = pipelines.get(change, pipelines['default'])
        return signal_message + build_overview(position, pipeline)

    return wrapper


# --------------------------------- Helper functions ---------------------------------
def _base_asset(symbol: str) -> str | None:
    if '/' in symbol:
        return symbol.split('/')[0]
    else:
        logger.warning(f"Unable to extract base asset from invalid symbol: {symbol}")
        return None


def _format_duration(seconds: int) -> str:
    """
    Convert duration from seconds to a human-readable format.

    :param seconds: Duration in seconds
    :return: Formatted string representing the duration
    """
    seconds = int(seconds)

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts[:-1]) if len(parts) > 1 else parts[0]


def _construct_buy_str(position: Dict[str, Any]) -> str:
    return f'aqucire more shares of {_base_asset(position.get("symbol"))}'


def _construct_sell_str(position: Dict[str, Any]) -> str:
    return f'sell some shares of {_base_asset(position.get("symbol"))}'


def _construct_change_position_message(
    action: str,
    change: str,
    position_type: str,
    change_percent: float,
) -> str:
    arrow = {
        ("*increasing*", "LONG"): UP_ARROW,
        ("*decreasing*", "LONG"): DOWN_ARROW,
        ("*increasing*", "SHORT"): DOWN_ARROW,
        ("*decreasing*", "SHORT"): UP_ARROW,
    }.get((change, position_type), "")

    return (
        f"{arrow} In the bustling market of memes and coins, it is my intention "
        f" to {action}, thereby {change} my {position_type} position by "
        f"not less than {round(change_percent, 3):g}%, my esteemed colleagues!"
    )


# ------------------------------------------------------------------------------------
def send_message(chat_id: str, msg: str) -> None:
    # Implementation of sending message to Telegram
    logger.info(f"Sending to {chat_id}: {msg}")

    send_text = "https://api.telegram.org/bot"
    send_text += bot_token + "/sendMessage?chat_id=" + chat_id
    send_text += f"&parse_mode=Markdown&text={msg}"

    try:
        response = requests.get(send_text)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
    except requests.exceptions.RequestException as e:
        logger.exception(f"Unable to send Telegram intro: {e}")
        logger.exception(
            f"Unable to send Telegram intro. "
            f"Response content: {e.response.content if e.response else 'No response'}"
        )


@add_position_overview
def construct_open_long_message(position: Dict[str, Any]) -> str:
    asset = position.get("symbol")

    return (
        f"{UP_ARROW} It is with great pleasure that I announce the commencement of my "
        f"involvement in the shares of {asset}, a noteworthy enterprise "
        "within the [industry/sector] sector. This investment, I do believe, "
        "it carries the prospect of a most advantageous return, and I eagerly "
        "look forward to monitoring its progress with keen interest."
        # "However, it must be noted that this missive is not intended to provide "
        # "financial counsel, and one is encouraged to seek the guidance of a "
        # "qualified financial advisor prior to any investment decision."
    )


@add_position_overview
def construct_close_long_message(position: Dict[str, Any]) -> str:
    asset = position.get("symbol")

    return (
        f"{DOWN_ARROW} It is with due consideration and deliberation that I have"
        f" elected to divest myself of my holdings in {asset}. \n\nWhilst the"
        " experience has proven both enlightening and profitable, I now find it "
        "prudent to  redirect my investments to other opportunities. It has been"
        " a privilege to partake in this endeavor, and I look forward to engaging"
        " in further ventures of a similar nature."
        # "As always, this communication serves as a "
        # "mere update and not as a source of financial guidance, and one is "
        # "advised to seek professional counsel in such matters."
    )


@add_position_overview
def construct_open_short_message(position: Dict[str, Any]) -> str:
    asset = position.get("symbol")

    return (
        f"{DOWN_ARROW} It is with a measure of calculated risk and astute observation"
        f"that I hereby declare my engagement in a short position regarding {asset}."
        "\n\nThis strategic maneuver, undertaken after thorough analysis and "
        "consideration, reflects my belief in a potential downturn in the "
        "asset's valuation. I shall monitor this position with utmost vigilance, "
        "prepared to adjust my strategy as market conditions dictate. \n\n"

        "As ever, one must approach such ventures with caution, for the "
        "markets are as unpredictable as they are enticing."
    )


@add_position_overview
def construct_close_short_message(position: Dict[str, Any]) -> str:
    asset = position.get("symbol")

    return (
        f"{UP_ARROW} I find myself compelled to announce, with a mixture of"
        f" satisfaction and prudence, the closure of my short position in {asset}. \n\n"
        "This speculative venture, undertaken with due diligence and foresight, "
        "has run its course. The vicissitudes of the market have proven "
        "most instructive, and I emerge from this engagement with both "
        "pecuniary gain and invaluable experience. \n\n"
        "As we retire from this particular arena of financial combat, I "
        "remain ever vigilant for future opportunities that may present "
        "themselves in this most unpredictable of theatres."
        # "I must reiterate, with the utmost clarity, that this missive "
        # "serves merely as a chronicle of my endeavors and not as financial "
        # "counsel. One would be well-advised to seek the guidance of a "
        # "qualified financial steward before embarking on similar ventures."
    )


@add_position_overview
def construct_increase_long_message(position: Dict[str, Any]) -> str:
    return _construct_change_position_message(
        _construct_buy_str(position),
        '*increasing*',
        'LONG',
        position.get('change_percent', 0)
        )


@add_position_overview
def construct_decrease_long_message(position: Dict[str, Any]) -> str:
    return _construct_change_position_message(
        _construct_sell_str(position),
        '*decreasing*',
        'LONG',
        position.get('change_percent', 0)
        )


@add_position_overview
def construct_increase_short_message(position: Dict[str, Any]) -> str:
    return _construct_change_position_message(
        _construct_sell_str(position),
        '*increasing*',
        'SHORT',
        position.get('change_percent', 0)
    )


@add_position_overview
def construct_decrease_short_message(position: Dict[str, Any]) -> str:
    return _construct_change_position_message(
        _construct_buy_str(position),
        '*decreasing*',
        'SHORT',
        position.get('change_percent', 0)
    )


@add_position_overview
def construct_do_nothing_message(position: Dict[str, Any]) -> str:
    return (
        "On this fine day, I shan't undertake any endeavours or pursuits.\n"
        "I shall simply partake in the leisurely art of relaxation, "
        "my good fellow!"
    )


def construct_error_message(*args) -> str:
    return (
        "Woe is me, for I have stumbled upon a veritable labyrinth of "
        "tribulations. The path forward is shrouded in obscurity, "
        "leaving me bereft of direction and at a profound loss for "
        "resolution. 😩"
        )


# ------------------------------------------------------------------------------------
def get_message_constructor(
    position: Dict[str, Any]
) -> Callable[[Dict[str, Any]], str]:
    """Returns a function that constructs a message based on the given position data.

    Arguments:
    ----------
        position (Dict[str, Any])
            The position data.
        chat_id (str)
            The chat ID for sending messages.
    Returns:
    -------
        Callable[[], None]
            A function that sends a message based on the given position data.
    """
    logger.debug(f"Creating message constructor for position: {position}")

    message_constructors = {
        ("open", "LONG"): construct_open_long_message,
        ("increase", "LONG"): construct_increase_long_message,
        ("decrease", "LONG"): construct_decrease_long_message,
        ("close", "LONG"): construct_close_long_message,
        ("open", "SHORT"): construct_open_short_message,
        ("increase", "SHORT"): construct_increase_short_message,
        ("decrease", "SHORT"): construct_decrease_short_message,
        ("close", "SHORT"): construct_close_short_message,
        (None, "LONG"): construct_do_nothing_message,
        (None, "SHORT"): construct_do_nothing_message,
    }
    return message_constructors.get(
        (position["change"], position["position_type"]),
        construct_error_message
        )


def create_signal(position: Dict[str, Any], chat_id: str) -> Callable[[], None]:
    if not (message_constructor := get_message_constructor(position)):
        raise ValueError(f"Invalid position data: {position}")

    return lambda: send_message(chat_id, message_constructor(position))


def send_intro(chat_id: str) -> None:
    intro = """Ahoy, lads and lasses! I be Gregorovich, once a human sailor and
now a ship mind aboard the Wallfish. Me words be naught but the whimsical
ramblings of a brain in a box, meant for your amusement only. *Nay, ye won't
find financial advice here, for me words be as untethered as me body.* So listen
up, and I'll spin ye a yarn, just don't be staking your fortune on it, ya hear?"""
    send_message(chat_id, intro)


def create_telegram_signal(
    position: Dict[str, Any], chat_id: str
) -> Dict[str, Callable[[], None]]:
    return {
        "send_intro": partial(send_intro, chat_id),
        "send_signal": create_signal(position, chat_id),
    }
