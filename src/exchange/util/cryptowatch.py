#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 23 11:07:23 2022

@author dhaneor
"""
import time
import cryptowatch as cw
import pandas as pd
import requests
import logging

from tenacity import retry
from tenacity.wait import wait_random_exponential, wait_exponential, wait_fixed
from tenacity.stop import stop_after_attempt
from tenacity.after import after_log
from typing import List, Optional


API_KEY = "Y7J7RL9050DIK0S5WX6Y"
SECRET = "RlwA5Yv08ap2vIAeOb6Xh5a/jaJcp6zFlWR4ZP0R"

PERIODS = {
            "1m": '60', 
            "3m": '180', 
            "5m": '300',
            "15m": '900',
            "30m": '1800', 
            "1h": '3600', 
            "2h": '7200',
            "4h": '14400',
            "6h": '21600', 
            "12h": '43200', 
            "1d": '86400',
            "3d": '259200',
            "1w": '604800'
}

logger = logging.getLogger('main.cryptowatch')
logger.setLevel(logging.INFO)
 
# =============================================================================
class CryptoWatch:
    """Class to interact with the Cryptowatch API"""
    
    def __init__(self):
        cw.api_key = API_KEY

        self._markets = []
        self._last_update_markets: int = 0
        self._cache_refresh_interval: int = 3600
        self.get_markets()
        
    @property
    def markets(self) -> list:
        def process_result(result):
            if data.get('result'):
                logger.info(data.get('allowance'))
                self._markets = data.get('result', [])
                self._last_update_markets = int(time.time())
            else:
                raise Exception('got no data')
        
        must_update_cache = (
            time.time() > self._last_update_markets \
                + self._cache_refresh_interval
        )
        
        if must_update_cache:
            logger.debug('updating markets from API')
            
            try:
                data = self.__download_markets()
                process_result(data)
            
            except Exception as e:
                
                if self._markets:
                    logger.error(
                        f'unable to get markets: {e} - will use cached data'
                    )
                    return self._markets
                
                else:
                    logger.critical(
                        f'unable to get markets: {e} - unable to continue ... for now'
                    )
                    
                    done = False
                    while not done:                        
                        try:
                            data = self.__download_markets()
                            process_result(data)
                            done=True
                        except Exception as e:
                            logger.critical(
                                f'this may take a very long time ({e}) ...'
                            )
                            time.sleep(3)

        return self._markets            
            
    # -------------------------------------------------------------------------
    def get_exchanges(self, return_only_active:bool=True) -> List[dict]:
        """Gets a list of all exchanges that Cryptowqatch tracks.

        :param return_only_active: return only active exchanges, 
        defaults to True
        :type return_only_active: bool, optional
        :return: list of dictionaries with exchange information
        :rtype: List[dict]
        """
        try:
            data = requests.get(
                'https://api.cryptowat.ch/exchanges', 
                params={'apikey' : API_KEY}
            ).json()
            
            res = item if (item := data.get('result')) else []
        except Exception as e:
            logger.error(e)
            return []
        
        if return_only_active:
            res = [item for item in data if item.get('active') == True]
        
        return res

    def get_markets(self, symbol: Optional[str]=None) -> List[dict]:
        """Gets a list of all markets (for one or all symbols).

        :param symbol: filter result by symbol name, defaults to None
        :type symbol: Union[str, None], optional
        :return: list of the markets
        :rtype: List[dict]
        """
        data = self.markets
                        
        if symbol:
            symbol = ''.join(symbol\
                .split('-')) if '-' in symbol else symbol\
                    .lower()            

        return [item for item in data if item.get('pair') == symbol]

    def get_ohlcv(self, exchange: str, symbol: str, 
                  interval: str) -> pd.DataFrame:
        """Gets OHLCV data for the given exchange/symbol/interval.
        
        The method attempts to download OHLCV data from the given 
        exchange. However, if this fails it tries to get the data
        for the symbol and interval, but from another exchange.
        This is not ideal, but better than no data at all.

        :param exchange: name of the exchange
        :type exchange: str
        :param symbol: name of the symbol (e.g. 'BTC-USDT' or 'BTCUSDT')
        :type symbol: str
        :param interval: the trading interval ('1m' .. '1d') 
        :type interval: str
        :return: the OHLCV data for the 1000 most recent intervals
        :rtype: pd.DataFrame
        """
        symbol = ''.join(symbol.split('-')) if '-' in symbol else symbol
        period = PERIODS.get(interval, '')
        
        # prepare a list of exchanges that we can use as data source
        valid_exchanges = self._get_exchanges_where_symbol_has_a_market(symbol)
        
        if not valid_exchanges:
            logger.error(f'could not find an exchange for {symbol}')
            return pd.DataFrame()

        liquid_exchanges = (
            'kucoin', 'binance', 'bitstamp', 'bitfinex', 'kraken', 'coinbase'
        )
                        
        exchanges = [exchange]
        
        [exchanges.append(exc) for exc in liquid_exchanges\
            if (exc in valid_exchanges) and not (exc in exchanges)]
       
        logger.debug(exchanges)
                
        # try to download until you get the data from any of the exchanges
        # will start with the one requested by the caller
        for exc in exchanges:        
            logger.info(f'downloading OHLCV ({symbol}) for: {exchange}')
            
            try:
                data = self.__download_ohlcv(exc, symbol, period)  
                
                ohlc = pd.DataFrame(
                    data=data['result'][period], 
                    columns=[
                        'close time', 
                        'open', 
                        'high', 
                        'low', 
                        'close', 
                        'volume', 
                        'quote volume'
                    ]
                )
                    
                return self._restructure_dataframe(ohlc)
            
            except Exception as e:
                logger.error(e)

        return pd.DataFrame()
            
    # -------------------------------------------------------------------------
    def _get_exchanges_where_symbol_has_a_market(self, symbol:str) -> List[str]:
        """Finds all exchanges where a symbol can be traded.

        :param symbol: name of the symbol (e.g. 'BTC-USDT')
        :type symbol: str
        :return: the names of the exchanges
        :rtype: List[str]
        """
        markets = self.get_markets(symbol=symbol)

        return [m['exchange'] for m in markets if m['active']] if markets else []

    def _restructure_dataframe(self, raw_df:pd.DataFrame) -> pd.DataFrame:
        """Adjusts the format of the DataFrame to our internal format.

        :param raw_df: _description_
        :type raw_df: pd.DataFrame
        :return: _description_
        :rtype: pd.DataFrame
        """
        interval_in_ms = raw_df.iat[1, 0] - raw_df.iat[0, 0]
        df = pd.DataFrame()
        
        df['open time'] = (raw_df['close time'] - interval_in_ms) * 1000
        df['human open time'] = pd.to_datetime(
            df['open time'], unit='ms', origin='unix'
        )
        df['open'] = raw_df['open'].astype(float)
        df['high'] = raw_df['high'].astype(float)
        df['low'] = raw_df['low'].astype(float)
        df['close'] = raw_df['close'].astype(float)
        df['close time'] = (df['open time'] + interval_in_ms - 1)
        df['volume'] = raw_df['volume'].astype(float)
        df['quote volumne'] = raw_df['quote volume'].astype(float)
        
        return df
    
    # -------------------------------------------------------------------------
    @retry(
        wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(3), 
        after=after_log(logger, logging.DEBUG),
        reraise=True
    )
    def __download_markets(self) -> dict:
        req = requests.get(
            'https://api.cryptowat.ch/markets', 
            params={'apikey' : API_KEY}
        )   
                
        return req.json()
    
    @retry(
        wait=wait_fixed(2),
        stop=stop_after_attempt(5),
        after=after_log(logger, logging.DEBUG),
        reraise=True
    )
    def __download_ohlcv(self, exchange:str, symbol:str, interval:str) -> dict:

        req = requests.get(
            'https://api.cryptowat.ch/markets/'+exchange+'/'+symbol+'/ohlc', 
            params={'periods': interval, 'apikey' : API_KEY}
        )
                
        return req.json()
        
        
        
    
    
    
"""
def main():
    exchange = 'binance'
    pair = 'ethusd'
    tf = '1m'
    before  = '2022-10-22 19:00:00'
    after = '2018-10-22 19:00:00'

    periods = {"1m": '60', 
            "3m": '180', 
            "5m": '300',
            "15m": '900',
            "30m": '1800', 
            "1h": '3600', 
            "2h": '7200',
            "4h": '14400',
            "6h": '21600', 
            "12h": '43200', 
            "1d": '86400',
            "3d": '259200',
            "1w": '604800'}

    period=periods.get(tf)

    req = requests.get('https://api.cryptowat.ch/markets/'+exchange+'/'+pair+'/ohlc', params={
        'periods': period,
        'before': (pd.to_datetime([before]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'),
        'after': (pd.to_datetime([after]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'),
        'apikey' : API_KEY
    })

    # pprint(req.__dict__)

    data = req.json()
    ohlc = pd.DataFrame(data['result'][period], 
                        columns=['close time', 
                                'open', 
                                'high', 
                                'low', 
                                'close', 
                                'volume', 
                                'quote volume'])
    ohlc['close time human'] = pd.to_datetime(ohlc['close time'], unit='s')
    # ohlc.drop('NA', axis=1)  

    print(ohlc)  
    
# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':
    main()
    
""" 