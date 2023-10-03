#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:55:20 2022

@author dhaneor
"""
import logging
import time
import pandas as pd

from threading import Thread
from multiprocessing import Queue
from collections import namedtuple
from typing import List, Tuple, Iterable, Union, Optional

from staff.moses import Moses
from broker.jupiter import Jupiter
from broker.models.symbol import Symbol
from models.users import Account
from analysis.util.ohlcv_observer import OhlcvObserver
from analysis.oracle import LiveOracle, OracleRequest
from analysis.strategies.definitions import strategy
from analysis.leverage import PositionSizeManager

"""
TODO    the current positions are still not shown correctly?!

TODO    implement a switch between binary/continuous trading!

TODO    implement a threshhold for minimal change in SL price,
        before the SL order is actually updated. This prevents
        changes that don´t make a difference and cost time

TODO    find a way to get the correct trailing SL prices from the
        DataFrame (-> groupby, see article). this will probably be
        faster and less error prone than getting the last SL price
        from the order that we get from the repository .. see also
        in positionhandler.py for more on this!

TODO    change Moses class so that the SL parameters are not set on
        the class level ... at least not only.
        There should be a way to provide them in the call to the
        method that calculates the SL prices so that every caller can
        have independent strategy and parameters!
"""

# =============================================================================
from analysis.strategies.definitions import STRATEGIES
from config import MIN_POS_CHANGE_PERCENT, ALLOW_POS_INCREASE

LOGGER_NAME = 'main.AccountManager'

ADJUST_FOR_SLIPPAGE = 0.99
PRECISION_LOG = 8
SIGNAL_COLUMN = 's.all'

logger = logging.getLogger(LOGGER_NAME)

# =============================================================================
position_request = namedtuple(
    'PositionRequest', ['symbol', 'side', 'leverage']
)

# type hint types
StopLossValues = Union[List[Tuple[float, float]], None]
TakeProfitValues = Union[List[Tuple[float, float]], None]

div = '~-*-~' * 10

# =============================================================================
class AccountManager(Thread):
    """Handles one user (sub-)account on the exchange.

    This class handles an account or sub-account for a user on the
    exchange. AccountManager subscribes with the OhlcvObserver for
    the symbols and in the interval/period that are configured for
    the given Account and acts upon Updates it receives from the
    OhlcvObserver.

    It asks for signals from the Oracle and then uses Jupiter (Execution
    Manager) to perform the necessary updates like open/close/update a
    position, stop loss or take profit.
    """

    def __init__(self, account: Account, ohlcv_observer: OhlcvObserver,
                 oracle: LiveOracle, notify_queue: Optional[Queue]=None,
                 dry_run: bool=False):
        """Initializes the AccountManager.

        :param account: user account object w/ account related parameters
        :type account: Account
        :param ohlcv_observer: the observer that calls back when new
        OHLCV data is available for the symbol(s) and interval that
        this account manager subscribed for
        :type ohlcv_observer: OhlcvObserver
        :param oracle: the Oracle that gives back the buy/sell signals
        :type oracle: LiveOracle
        """
        super().__init__(daemon=True)
        self.account: Account = account
        self.id: str = f'{str(account.id)}({account.name})'

        self.strategies = STRATEGIES

        self.logger = logging.getLogger(f'{LOGGER_NAME}.{self.id}')

        self.ohlcv_observer: OhlcvObserver = ohlcv_observer
        self.oracle: LiveOracle = oracle
        self.position_size_manager: PositionSizeManager = \
            PositionSizeManager()
        self.stop_loss_manager = Moses()

        self.execution_manager = Jupiter(
            exchange=account.exchange, # type: ignore
            market=account.market, # type: ignore
            user_account=self._get_credentials()
        )

        self.broker = self.execution_manager.broker
        self.notify_queue = notify_queue

        self.quote_asset = account.quote_asset
        self.execution_manager.quote_asset = account.quote_asset # type: ignore

        self.number_of_assets: int
        self.current_positions: Iterable
        self.position_requests: list
        self.leverage_values: list
        self.epoch: int = 1
        self.execution_time: float

        self._subscribe_to_ohlcv_observer()

        self.logger.info(f'account manager initialized for {account}')

    @property
    def currently_active_assets(self) -> List[str]:
        assets = []
        for s in self.strategies:
            symbol = self.broker.get_symbol(s.symbol)
            if symbol:
                assets.append(symbol.base_asset)

        return assets

    # -------------------------------------------------------------------------
    def handle_ohlcv_update(self, message:dict):
        """The main entry point when new OHLCV data is available.

        This method is given to the OhlcvObserver as callback to send
        new OHLCV data to.

        :param message: the message from OhlcvObserver
        :type message: dict

        .. code:: python
        {
            'id': <hopefully the id of this account manager>,
            'data':
                {
                    'symbol': <name of the symbol>,
                    'interval': <name of the interval>,
                    'data': <a DataFrame with OHLCV data>
                },
                {
                    ...
                }

        }
        """
        start_time = time.time()
        self.logger.debug(f'{div} got an update {div}')

        # make sure that we got data and that it is for the coins that
        # this account intends to trade
        if not message:
            self.logger.critical('got empty data')
            return

        if message['id'] != self.id:
            id_ = message['id']
            self.logger.error(f'got wrong data! it´s for someone else {id_}!')
            return

        # in case the user manually opened positions in the account or
        # sub-account that we are trading in ... check for those and
        # just close them
        self._close_ghost_positions()

        # .....................................................................
        data = message['data']
        self.position_requests, self.leverage_values = [], []
        self.notional_exposure, self.diversification_multiplier = 0, 1
        self.number_of_assets = min(len(self.strategies), len(data))

        # make sure that at least one strategy is set and that we actually
        # got ohlcv data
        if self.number_of_assets == 0:
            self.logger.critical(f'got no data?')
            return

        # the diversification multiplier is set dynamically (depending
        # on the correlation of our assets) and allows us to have bigger
        # position sizes when correlation is low
        self._determine_diversification_multiplier(data)

        for item in data:
            try:
                self._process_data_for_one_symbol(item)
            except Exception as e:
                s = item.get('symbol', 'UNKNOWN')
                self.logger.critical(
                    f'processing data for {s} failed:'
                )
                self.logger.exception(msg=e, stack_info=True)

        # in case that the previous step ended with an exception for
        # all our assets, it makes no sense to continue. this should
        # never happen, though!
        if not self.position_requests:
            self.logger.critical('no position requests - I´m outta here!')
            return

        # .....................................................................
        # In case the combined position sizes are higher than we can
        # afford to trade, scale them down ... this is an edge case
        # that should also never happen. But it is better to check
        # this, than to risk API errors which would need to be handled.
        max_exposure = self._get_account_value() * \
            min(
                self._get_max_leverage_allowed_by_exchange(),
                self._get_max_leverage_allowed_by_user()
            )

        self.logger.info(
            f'planned notional exposure: '\
                f'{round(self.notional_exposure, PRECISION_LOG)} '\
                    f'(max: {round(max_exposure, PRECISION_LOG)})'
        )

        if self.notional_exposure > max_exposure:
            overshoot = self.notional_exposure / max_exposure
            self.notional_exposure = 0

            [self._adjust_target_value_for_request(req, overshoot) \
                for req in self.position_requests]

            self.logger.info(
                f'planned notional exposure: '\
                    f'{round(self.notional_exposure, PRECISION_LOG)} '\
                        f'(max: {max_exposure})'
            )

        # .....................................................................
        # Let our Execution Manager execute all necessary actions,
        # required to set our positions to the targets that we just
        # determined.
        self.execution_manager.update_account(self.position_requests)
        execution_time = round(time.time() - start_time, 3)

        self.logger.info(self.execution_manager.account)
        self.logger.info(
            f'epoch {self.epoch} - execution time: ' \
                f'{execution_time} seconds'
        )

        # .....................................................................
        # Send a summary over our notify queue and let whoever is interested
        # deal with this
        if self.notify_queue:
            self.notify_queue.put(self.execution_manager.account.get_summary())

        self.epoch += 1

    # -------------------------------------------------------------------------
    def _process_data_for_one_symbol(self, item: dict):
        symbol, interval = item.get('symbol', ''), item.get('interval', '')
        ohlcv_df = item['data']
        self.proceed = True

        l = '~' * 50
        self.logger.info(
            f'{l} processing data for: {symbol} ({len(ohlcv_df)}) {l}'
        )

        # TODO  validate input?

        # .....................................................................
        ohlcv_df, strategy, request = ohlcv_df.copy(deep=True),  None, {}
        symbol_obj = self.broker.get_symbol(symbol)

        if isinstance(symbol_obj, Symbol):
            request['symbol'] = symbol_obj
            request['asset'] = symbol_obj.base_asset
            request['quote_asset'] = symbol_obj.quote_asset
        else:
            raise TypeError(f'symbol_obj is: {type(symbol_obj)} for {symbol}')

        # .....................................................................
        for s in self.strategies:
            if s.symbol == symbol and s.interval == interval:
                strategy = s

        if not strategy:
            self.logger.error(f'found no strategy for: {symbol}, {interval}')
            return

        # .....................................................................
        # get the trading signal from our oracle
        signal, ohlcv_df = self._ask_oracle(
            symbol=symbol, interval=interval, strategy=strategy.name,
            df=ohlcv_df
        )

        self.logger.debug(f'got signal: {signal}')

        if signal == 0:
            self.logger.info(ohlcv_df.tail(5))
            self.logger.info('nothing to do here ...')
            return

        # .....................................................................
        # determine leverage and the resulting position size
        leverage = self._get_leverage(df=ohlcv_df, signal=signal)
        price = ohlcv_df.iloc[-1].loc['close']

        position_size = self._get_position_size(
            price=price,
            leverage=leverage,
            signal=signal,
            asset=symbol_obj.base_asset
        )
        request['target'] = position_size
        request['notional'] = self._get_notional_size(req=request, price=price)

        # .....................................................................
        # determine stop loss and take profit (not implemented yet) values
        request['stop_loss'] = self._get_stop_losses(
            df=ohlcv_df, pos_size=position_size, strategy=strategy
        )
        request['take_profit'] = None

        # log everything
        self.logger.info(f'\n{ohlcv_df.tail(5)}')
        self.logger.info(f'target leverage for {symbol}: {round(leverage, 2)}x')
        self.logger.info(request)

        # .....................................................................
        del request['symbol']
        self.position_requests.append(request)

    # -------------------------------------------------------------------------
    def _ask_oracle(self, symbol: str, interval: str, strategy: str,
                    df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:

        oracle_request = OracleRequest(
            symbol=symbol, interval=interval, strategy=strategy, data=df
        )

        df = self.oracle.speak(request=oracle_request)
        signal = self._get_signal_from_df(df=df)

        return signal, df

    def _get_signal_from_df(self, df: pd.DataFrame) -> float:
        try:
            signal = df.iloc[-1].loc[SIGNAL_COLUMN]
        except Exception as e:
            self.logger.exception(e)
            signal = 0
            self.proceed = False

        return signal

    # .........................................................................
    def _get_leverage(self, df: pd.DataFrame, signal: float) -> float:

        df = self.position_size_manager.get_max_leverage(
            df=df, risk_level=self.account.risk_level # type: ignore
        )

        requested_leverage = df.iloc[-1].loc['leverage'] \
            * self.diversification_multiplier * abs(signal)

        max_leverage_allowed_by_user = self._get_max_leverage_allowed_by_user()
        max_leverage_on_exchange = self._get_max_leverage_allowed_by_exchange()

        if signal < 0:
            max_leverage_on_exchange -= 1

        max_leverage = min(
            max_leverage_allowed_by_user, max_leverage_on_exchange
        )

        self.leverage_values.append(max_leverage)

        return min(requested_leverage, max_leverage)

    def _get_max_leverage_allowed_by_user(self) -> float:
        ml = self.account.max_leverage
        if isinstance(ml, Iterable):
            return float(ml[0]) # type: ignore
        else:
            return float(ml) # type: ignore

    def _get_max_leverage_allowed_by_exchange(self):

        market = self.broker.market

        if market.upper() == 'SPOT':
            return 1

        elif market.upper() == 'CROSS MARGIN':
            margin_config = self.broker.margin_configuration
            return margin_config['maxLeverage'] * ADJUST_FOR_SLIPPAGE

        else:
            self.logger.error(
                f'unable to determine max leverage on exchange for market '\
                    f'{market} - disabling leveraged trading now!'
            )
            return 1

    # .........................................................................
    def _get_position_size(self, price: float, leverage: float, signal: float,
                           asset: str) -> float:

        max_notional_size = self._get_account_value() / self.number_of_assets

        notional_size = max_notional_size / price * leverage
        position_size = notional_size * ADJUST_FOR_SLIPPAGE

        if signal < 0:
            position_size *= -1

        self.logger.debug(
            f'p.size: {round(max_notional_size/price, PRECISION_LOG)} * '\
                f'{round(leverage, PRECISION_LOG)}'\
                    f' * {ADJUST_FOR_SLIPPAGE} = '\
                        f'{round(position_size, PRECISION_LOG)}'
        )

        if ALLOW_POS_INCREASE:
            return position_size

        # If ALLOW_POS_INCREASE is set to False, this checks if the requested
        # position size would exceed the current position size - adjusts new
        # value if necessary.
        try:
            curr_pos_type, curr_balance = self._get_current_position(asset)

            self.logger.debug(f'{curr_pos_type} -> balance: {curr_balance}')

            if curr_balance is None:
                return position_size

            if abs(position_size) > abs(curr_balance):
                return curr_balance
            else:
                return position_size

        except Exception as e:
            self.logger.exception(e)

        return 0

    def _get_stop_losses(self, df: pd.DataFrame, pos_size: float,
                         strategy: strategy) -> StopLossValues:

        if not strategy.sl_strategy:
            return None

        self.stop_loss_manager.set_sl_strategy(
            strategy=strategy.sl_strategy, sl_params=strategy.sl_params
        )

        ohlcv_df = self.stop_loss_manager.get_stop_loss_prices(df=df)

        stop_price = None

        if pos_size > 0:
            stop_price = ohlcv_df.iloc[-1].loc['sl.long']
        elif pos_size < 0:
            stop_price = ohlcv_df.iloc[-1].loc['sl.short']

        if not stop_price:
            return None

        # .....................................................................
        active_sl = self.broker.get_active_stop_orders(strategy.symbol)
        o = active_sl[-1] if active_sl else None
        [self.logger.debug(o.stop_price) for o in active_sl]

        # in case there is no currently active SL order
        if not o:
            return [(stop_price, 1)]if stop_price else None

        # in case there is an active SL order
        prev_stop_price = o.stop_price

        if pos_size > 0:
            if o.side == 'SELL' and (stop_price < prev_stop_price):
                stop_price = prev_stop_price

        elif pos_size < 0:
            if o.side == 'BUY' and (stop_price > prev_stop_price):
                stop_price = prev_stop_price

        return [(stop_price, 1)]

    def _get_notional_size(self, req: dict, price: float) -> dict:
        req['target'] = round(req['target'], 9)
        req = self._validate_request(request=req)

        target = req['target']
        notional = abs(target) * price
        self.notional_exposure += notional
        precision = req['symbol'].quote_asset_precision

        return round(notional, precision) # type: ignore

    def _adjust_target_value_for_request(self, req: dict, factor: float):
        last_price = self.broker.get_last_price(req['symbol'])

        req['target'] /= factor
        req = self._validate_request(req)

        target = req['target']
        notional = abs(target) * last_price

        req['notional'] = notional
        self.notional_exposure += notional

        self.logger.debug(f'{target=}, {last_price=} -> {abs(notional)}')
        self.logger.info(f'final position request: {req}')

    # -------------------------------------------------------------------------
    def _close_ghost_positions(self):
        acc = self.execution_manager.account
        current_positions = acc.get_all_positions(
            include_dust=False, update=False
        )

        self.current_positions = current_positions
        self._log_current_positions(current_positions)

        current_positions = (
            pos for pos in current_positions if pos.asset != self.quote_asset
        )

        dont_touch = self.currently_active_assets + ['KCS']

        for asset in (pos.asset for pos in current_positions):
            if asset not in dont_touch:
                acc.reset_account(dont_touch=dont_touch)
                return

    def _log_current_positions(self, current_positions: Iterable[object]):
        self.logger.info('these are our current positions:')

        for pos in current_positions:
            if not pos.is_dust: # type: ignore
                self.logger.info(f'position: {pos}')
            else:
                self.logger.info(f'no position: {pos}')

    def _log_margin_loan_info(self, symbol: Symbol):
        margin_loan_info = self.broker.get_margin_loan_info(symbol.base_asset)
        self.logger.debug(margin_loan_info)

        margin_loan_info = self.broker.get_margin_loan_info(symbol.quote_asset)
        self.logger.debug(margin_loan_info)

    def _get_current_position(
        self,
        asset: str
        ) -> Tuple[Union[str, None], Union[float, None]]:

        position = self.execution_manager.account.get_position(asset)

        if position and position.balance.net > 0 and not position.is_dust:
            return 'LONG',position.balance.net
        elif position and position.balance.net < 0 and not position.is_dust:
            return 'SHORT', position.balance.net
        elif position and position.is_dust:
            return None, None
        else:
            raise Exception('unable to determine current position')

    def _get_account_value(self) -> float:
        account = self.execution_manager.get_account()
        return account.get_account_value()

    def _validate_request(self, request: dict):
        symbol = request['symbol']
        request_target = request['target']
        current_balance = self.broker.get_balance(symbol.base_asset)

        if not current_balance:
            self.logger.error(
                f'unable to determine current balance for {symbol}! '\
                    f'will try to close position for security reasons ...'
            )
            request['target'] = 0
            return request

        current_net_balance = current_balance['net']

        # .....................................................................
        change = abs(current_net_balance - request_target)
        dust = symbol.lot_size_min
        change_min = max(
            abs(current_net_balance * (MIN_POS_CHANGE_PERCENT / 100)), dust
        )

        if (change < dust) or (change < change_min):
            self.logger.info(
                f'{symbol.name} position change request declined - '\
                f'target: {round(request_target, PRECISION_LOG)} :: '\
                f'current: {round(current_net_balance, PRECISION_LOG)} '\
                f'change: {round(change, PRECISION_LOG)} :: '\
                f'change min: {round(change_min, PRECISION_LOG)} ({dust})'
            )

            request['target'] = current_net_balance

        return request

    # -------------------------------------------------------------------------
    def _get_credentials(self):
        return {
            'api_key': self.account.api_key,
            'api_secret': self.account.api_secret,
            'api_passphrase': self.account.api_passphrase
        }

    def _subscribe_to_ohlcv_observer(self):

        symbols = tuple(strat.symbol for strat in self.strategies)

        interval = tuple(
            set(
                (strat.interval for strat in self.strategies)
            )
        )[0]

        self.logger.info(
            f'subscribing with OHLCV Observer for {symbols} ({interval})'
        )

        self.ohlcv_observer.register_subscriber(
            id=self.id,
            symbols=symbols,
            interval=interval,
            callback=self.handle_ohlcv_update
        )

    def _determine_diversification_multiplier(self, data: Iterable[dict]):
        self.diversification_multiplier = self.position_size_manager\
            .get_diversification_multiplier(data=data, lookback=30)

        self.logger.info(
            f'diversification multiplier: {self.diversification_multiplier}'
        )
