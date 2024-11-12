#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 11 22:30:20 2024

@author dhaneor
"""
import logging
import json
import requests

logger = logging.getLogger('main.telegram_signal')


class TelegramSignal:
    def __init__(self, current_position: dict | None, chat_id: str = None):
        self.current_position = current_position

        self.bot_token = '1406970784:AAGB7t29hz9DR2X8SMtYFfvet2GcKTNO12w'
        self.bot_chat_id = chat_id or '600751557'

        self.base_msg = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        self.msg_attrs = f'?{self.bot_chat_id}=&parse_mode=Markdown&text='

    def get_message(self) -> str:
        """Builds the message from the current position."""
        if not self.current_position:
            return (
                "Waiting for the next opportunity. \nChill, bro!"
                )

        logger.debug(self.current_position)

        position_type = self.current_position.get('position_type', 'Unknown')
        required_action = self.current_position.get('required_action', None)

        action_str = (
            F"*{self.current_position.get('change').capitalize()} position by* "
            f"`{self.current_position.get('change_percent'):.4f}%`!\n"
            f"*Target Leverage*: `{self.current_position.get('leverage'):.4f}`\n"
        ) if required_action else ""

        required_action = required_action or "Chill, bro!"

        if (sl := self.current_position.get('stop_loss')):
            stop_loss = f"\nStop Loss: {sl}"
        else:
            stop_loss = ""

        if (tp := self.current_position.get('take_profit')):
            take_profit = f"\nTake Profit: {tp}"
        else:
            take_profit = ""

        msg = (
            f"*{self.current_position.get('symbol')} {position_type}* position "
            f"since {self.current_position.get('entry_time')}\n\n"
            f"Entry Price: `{self.current_position.get('entry_price')}`\n"
            f"Current Price: `{self.current_position.get('current_price')}`\n"
            f"PNL: `{round(self.current_position.get('pnl_percentage'), 2)}%`\n\n"
            f"*Action: {required_action}*\n"
            f"{action_str}{stop_loss}{take_profit}"

        )

        return msg

    def send_signal(self):

        message = self.get_message()

        send_text = 'https://api.telegram.org/bot'
        send_text += self.bot_token + '/sendMessage?chat_id=' + self.bot_chat_id
        send_text += f'&parse_mode=Markdown&text={message}'

        try:
            response = requests.get(send_text)
            response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        except requests.exceptions.RequestException as e:
            logger.exception(f"Unable to send Telegram message: {e}")
