#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on Nov 11 22:30:20 2024

@author dhaneor
"""
import logging
import os

from functools import wraps
from io import BytesIO
from telegram import Bot
from typing import Callable, Dict, Any, List, Awaitable

logger = logging.getLogger('main.telegram_signal')

BOT_TOKEN = os.getenv("BOT_TOKEN")
TG_API_URL = "https://api.telegram.org/bot"
PARSE_MODE = "Markdown"

UP_ARROW = "\U0001F4C8"  # "\U0001F53C"  # 🔼
DOWN_ARROW = "\U0001F4C9"  # "\U0001F53D"  # 🔽

# Initialize Telegram bot
bot = Bot(token=BOT_TOKEN)


# ------------------------------------------------------------------------------------
# async def send_message(chat_id: str, msg: str) -> None:
#     try:
#         await bot.send_message(chat_id=chat_id, text=msg, parse_mode=PARSE_MODE)
#         logger.info(f"Message sent successfully to chat {chat_id}")
#     except Exception as e:
#         logger.error(f"Error sending message: {e}")


async def send_message(
    chat_id: str,
    msg: str,
    image: BytesIO = None
) -> None:
    try:
        if image:
            # If an image is provided, send it along with the message
            image.seek(0)  # Ensure we're at the start of the BytesIO object
            await bot.send_photo(
                chat_id=chat_id,
                photo=image,
                caption=msg,
                parse_mode=PARSE_MODE
            )
        else:
            # If no image, send text message only
            await bot.send_message(
                chat_id=chat_id,
                text=msg,
                parse_mode=PARSE_MODE
            )

        logger.info(
            f"Message {'with picture' if image else ''} sent "
            f"successfully to chat {chat_id}"
            )
    except Exception as e:
        logger.error(
            f"Error sending message {'with picture' if image else ''}: {e}",
            exc_info=True
            )
        logger.error("Token: %s / Chat ID: %s", BOT_TOKEN, chat_id)


# ------------------------------------------------------------------------------------
def position_overview(func):
    """Wrapper function to add position overview to the message."""
    # helper functions for formatting values
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

    # helper functions to build the overview message
    def entry_time(position: Dict[str, Any]) -> str:
        return f"*Entry:*  \t{fv(position.get('entry_time'))}\n"

    def duration(position: Dict[str, Any]) -> str:
        return f"*Duration:*  \t{fv(_format_duration(position.get('duration')))}\n"

    def entry_price(position: Dict[str, Any]) -> str:
        return f"\n*Entry Price:   *  \t{fv(position.get('entry_price'))}\n"

    def current_price(position: Dict[str, Any]) -> str:
        return f"*Current Price: * {fv(position.get('current_price'))}\n\n"

    def pnl(position: Dict[str, Any]) -> str:
        return f"*Profit:       *   {fv(position.get('pnl_percentage'))}`%`\n"

    def max_drawdown(position: Dict[str, Any]) -> str:
        return f"*Drawdown:* {fv((position.get('max_drawdown') * 100))}`%`\n\n"

    def exit_price(position: Dict[str, Any]) -> str:
        return f"*Exit Price:* {fv(position.get('exit_price'))}\n\n"

    def leverage(position: Dict[str, Any]) -> str:
        return f"*Leverage:* {fv(position.get('leverage'))}\n"

    # function pipeline for each 'change' type for use in the subsequent
    # build_overview function
    pipelines = {
        "open": [
            entry_time, entry_price, leverage
            ],
        "close": [
            pnl, max_drawdown, entry_time, duration, entry_price, exit_price,
            ],
        "increase": [
            pnl, max_drawdown, leverage,
            entry_time, duration, entry_price, current_price,
            ],
        "decrease": [
            pnl, max_drawdown, leverage,
            entry_time, duration, entry_price, current_price,
            ],
        "default": [
            pnl,  max_drawdown, entry_time, duration, entry_price, current_price,
            ]
    }

    # main function to build the overview message
    def build_overview(position: Dict[str, Any], pipeline: List[Callable]) -> str:
        return (
            f"\n\n*{position.get('symbol')} - "
            f"{position.get('position_type')} position*\n"
            + "".join(func(position) for func in pipeline)
        )

    # wrapper function for using this as a decorator
    @wraps(func)
    async def wrapper(position: Dict[str, Any] | None) -> str:
        signal_message = await func(position)

        # No need to build an overview if we have no position
        if position is None:
            return signal_message

        # determine the appropriate pipeline based on the 'change' type
        change = position.get('change', 'default')
        pipeline = pipelines.get(change, pipelines['default'])

        return signal_message + build_overview(position, pipeline)

    return wrapper


# --------------------------------- Helper functions ---------------------------------
async def _base_asset(symbol: str) -> str | None:
    if '/' in symbol:
        return symbol.split('/')[0]
    else:
        logger.warning(f"Unable to extract base asset from invalid symbol: {symbol}")
        return None


async def _build_buy_str(position: Dict[str, Any]) -> str:
    return f'aqucire more shares of {await _base_asset(position.get("symbol"))}'


async def _build_sell_str(position: Dict[str, Any]) -> str:
    return f'sell some shares of {await _base_asset(position.get("symbol"))}'


async def _build_change_position_message(
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
@position_overview
async def build_open_long_message(position: Dict[str, Any]) -> str:
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


@position_overview
async def build_close_long_message(position: Dict[str, Any]) -> str:
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


@position_overview
async def build_open_short_message(position: Dict[str, Any]) -> str:
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


@position_overview
async def build_close_short_message(position: Dict[str, Any]) -> str:
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


@position_overview
async def build_increase_long_message(position: Dict[str, Any]) -> str:
    return _build_change_position_message(
        await _build_buy_str(position),
        '*increasing*',
        'LONG',
        position.get('change_percent', 0)
        )


@position_overview
async def build_decrease_long_message(position: Dict[str, Any]) -> str:
    return _build_change_position_message(
        await _build_sell_str(position),
        '*decreasing*',
        'LONG',
        position.get('change_percent', 0)
        )


@position_overview
async def build_increase_short_message(position: Dict[str, Any]) -> str:
    return _build_change_position_message(
        await _build_sell_str(position),
        '*increasing*',
        'SHORT',
        position.get('change_percent', 0)
    )


@position_overview
async def build_decrease_short_message(position: Dict[str, Any]) -> str:
    return _build_change_position_message(
        await _build_buy_str(position),
        '*decreasing*',
        'SHORT',
        position.get('change_percent', 0)
    )


@position_overview
async def build_do_nothing_message(position: Dict[str, Any] | None) -> str:
    return (
        "On this fine day, I shan't undertake any endeavours or pursuits.\n"
        "I shall simply partake in the leisurely art of relaxation, "
        "my good fellow!"
    )


async def build_wait_for_opportunity_message(*args, **kwargs) -> str:
    wait_str = (
        "\n\nUnder the current market conditions, I find myself"
        " unable to find any suitable opportunities. I shall continue to "
        "awaiting for the next opportunity that may present itself."
        )
    return await build_do_nothing_message(None) + wait_str


async def build_error_message(*args) -> str:
    return (
        "Woe is me, for I have stumbled upon a veritable labyrinth of "
        "tribulations. The path forward is shrouded in obscurity, "
        "leaving me bereft of direction and at a profound loss for "
        "resolution. 😩"
        )


# ------------------------------------------------------------------------------------
async def get_message_constructor(
    position: Dict[str, Any] | None
) -> Callable[[Dict[str, Any]], Awaitable[str]]:
    logger.debug(f"Creating message constructor for position: {position}")

    if position is None:
        return build_wait_for_opportunity_message  # default/fallback function

    message_constructors = {
        ("open", "LONG"): build_open_long_message,
        ("increase", "LONG"): build_increase_long_message,
        ("decrease", "LONG"): build_decrease_long_message,
        ("close", "LONG"): build_close_long_message,
        ("open", "SHORT"): build_open_short_message,
        ("increase", "SHORT"): build_increase_short_message,
        ("decrease", "SHORT"): build_decrease_short_message,
        ("close", "SHORT"): build_close_short_message,
        (None, "LONG"): build_do_nothing_message,
        (None, "SHORT"): build_do_nothing_message,
    }

    return message_constructors.get(
        (position["change"], position["position_type"]),
        build_error_message   # default/fallback function
    )


async def create_signal(position: Dict[str, Any]) -> Callable[[], None]:
    message_constructor = await get_message_constructor(position)
    return await message_constructor(position)


async def send_intro(chat_id: str) -> None:
    intro = """Ahoy, lads and lasses! I be Gregorovich, once a human sailor and
now a ship mind aboard the Wallfish. Me words be naught but the whimsical
ramblings of a brain in a box, meant for your amusement only. *Nay, ye won't
find financial advice here, for me words be as untethered as me body.* So listen
up, and I'll spin ye a yarn, just don't be staking your fortune on it, ya hear?"""
    await send_message(chat_id, intro)
