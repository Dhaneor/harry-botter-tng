#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 07 00:55:53 2022

@author: dhaneor
"""

import abc 
from typing import Iterable, Tuple, List, Union, Optional, Dict 

# =============================================================================
class IExchangePublic(abc.ABC):
    
    '''Methods for the exchange class that use the public API and general 
        methods, that are independent from the (type of) API'''

    def init(self):
        self.repository: object

    # -------------------------------------------------------------------------    
    @abc.abstractmethod
    def get_server_time(self) -> Union[int, None]:
        pass
    
    @abc.abstractmethod
    def get_server_status(self) -> Union[dict, None]:
        pass
    
    @abc.abstractmethod
    def get_markets(self):
        pass    

    @abc.abstractmethod
    def get_currencies(self) -> List[dict]:
        """Gets a tuple with all currencies/coins that can be traded.
        :return: {'success' : True,
                        'message' : result,
                        'warning' : warning,
                        'error' : None,
                        'error code' : None,
                        'status code' : 200,
                        'execution time' :round((time() - _st) * 1000)
                        }
        
        .. code::python
        # result is hopefully:
        [
            {
                "currency": "BTC",
                "name": "BTC",
                "fullName": "Bitcoin",
                "precision": 8,
                "withdrawalMinSize": "0.002",
                "withdrawalMinFee": "0.0005",
                "isWithdrawEnabled": true,
                "isDepositEnabled": true,
                "isMarginEnabled": true,
                "isDebitEnabled": true
            }, {

                "currency": "ETH",
                "name": "ETH",
                "fullName": "Ethereum",
                "precision": 8,
                "withdrawalMinSize": "0.02",
                "withdrawalMinFee": "0.01",
                "isWithdrawEnabled": true,
                "isDepositEnabled": true,
                "isMarginEnabled": true,
                "isDebitEnabled": true
                }
            ]
        :rtype: Tuple[dict]
        """
        pass
        
    @abc.abstractmethod
    def get_symbols(self) -> Iterable[dict]:
        pass
    
    @abc.abstractmethod
    def get_symbol(self, symbol: str) -> Union[dict, None]:
        pass
    
    @abc.abstractmethod
    def get_ticker(self, symbol: str) -> Union[dict, None]:
        pass
    
    @abc.abstractmethod
    def get_all_tickers(self) -> Tuple[dict]:
        pass
    
    @abc.abstractmethod
    def get_ohlcv(self, symbol: str, interval: str, 
                  start: Union[int, str, None]=None, 
                  end: Union[int, str, None]=None, 
                  as_dataframe: bool=True) -> dict:
        pass
    
    # -------------------------------------------------------------------------
    @abc.abstractmethod
    def _get_earliest_valid_timestamp(self):
        pass

 
# =============================================================================
class IExchangeTrading(abc.ABC):
    @abc.abstractmethod
    def get_account(self) -> Union[Tuple[dict], None]:
        pass   
    
    @abc.abstractmethod
    def get_debt_ratio(self) -> Union[float, None]:
        pass
    
    @abc.abstractmethod
    def get_balance(self) -> dict:
        pass 
    
    @abc.abstractmethod
    def get_fees(self) -> list:
        pass
        
    # .........................................................................
    @abc.abstractmethod
    def get_orders(self, symbol: Optional[str]=None, side: Optional[str]=None, 
                   order_type: Optional[str]=None,
                   status: Optional[str]=None, start:Union[int, str, None]=None, 
                   end:Union[int, str, None]=None) -> Union[Tuple[dict], None]:
        """Gets multiple orders, filtered if requested.

        :param symbol: name of the symbol, defaults to None
        :type symbol: str, optional
        :param side: BUY or SELL, defaults to None
        :type side: str, optional
        :param order_type: MARKET, LIMIT, ..., defaults to None
        :type order_type: str, optional
        :param status: NEW, FILLED, CANCELED, defaults to None
        :type status: str, optional
        :param start: start date, defaults to None
        :type start: Union[int, str], optional
        :param end: end date, defaults to None
        :type end: Union[int, str], optional
        :return: list of orders (order: dictionary)
        :rtype: List[dict]
        """
        pass
    
    @abc.abstractmethod
    def get_order(self, order_id: Optional[str]=None, 
                  client_order_id: Optional[str]=None) -> dict:
        """Gets information on one specific order.
        
        Either order id or client order id must be provided.

        :param order_id: order is as set by the exchange, defaults to None
        :type order_id: Optional[str], optional
        :param client_order_id: order id set by client (or random)
        :type client_order_id: Optional[str], optional
        :return: _description_
        :rtype: dict
        
        .. code::python
        
        # should provide the following fields in 'message'
        {
            'clientOrderId': r['clientOid'],
            'cummulativeQuoteQty': '0',
            'executedQty': '0',
            'fills': [],
            'icebergQty': '0.00000000',
            'isWorking': True if r['status'] == 'NEW' else False,
            'orderId': r['id'],
            'orderListId': -1,
            'origQty': r['size'],
            'origQuoteOrderQty': orig_quote_qty,
            'price': r['price'],
            'side': r['side'].upper(),
            'status': r['status'],
            'stopPrice': r['stopPrice'],
            'symbol': r['symbol'],
            'time': int(r['orderTime']/1_000_000),
            'timeInForce': r['timeInForce'],
            'type': type,
            'updateTime': int(r['orderTime']/1_000_000)
        }
        """
        pass
        
    @abc.abstractmethod
    def get_active_orders(self, symbol: Optional[str]=None, 
                          side: Optional[str]=None, 
                          ) -> Union[Tuple[dict], None]:
        pass

    @abc.abstractmethod
    def get_active_stop_orders(self, symbol: Union[str, None]=None
                               ) -> Union[Tuple[dict], None]:
        pass
    
    # .........................................................................
    @abc.abstractmethod
    def buy_market(self, symbol: str, client_order_id: Optional[str]=None, 
                   base_qty: Optional[float]=None, 
                   quote_qty: Optional[float]=None, 
                   auto_borrow=False) -> Union[dict, None]:
        pass
    
    @abc.abstractmethod
    def sell_market(self, symbol: str, client_order_id: Optional[str]=None, 
                    base_qty: Optional[float]=None, 
                    quote_qty: Optional[float]=None, 
                    auto_borrow=False) -> Union[dict, None]:
        pass

    @abc.abstractmethod
    def buy_limit(self, symbol: str, price: str, base_qty: Optional[str]=None, 
                  client_order_id: Optional[str]=None, margin_mode: str='cross', 
                  auto_borrow: bool=False, stp: Optional[str]=None, 
                  remark: Optional[str]=None) -> Union[dict, None]:
        pass

    @abc.abstractmethod
    def sell_limit(self, symbol: str, price: str, base_qty: Optional[str]=None, 
                   client_order_id: Optional[str]=None, 
                   margin_mode: str='cross',
                   auto_borrow: bool=False, stp: Optional[str]=None, 
                   remark: Optional[str]=None) -> Union[dict, None]:
        pass

    @abc.abstractmethod
    def stop_limit(self, symbol:str, side: str, base_qty: str, 
                   stop_price: str, limit_price: str,
                   client_order_id: Optional[str]=None, 
                   loss_or_entry: str='loss',
                   ) -> Union[dict, None]:
        pass
    
    
    @abc.abstractmethod
    def stop_market(self, symbol: str, side: str, base_qty: str, 
                    stop_price: str, client_order_id: Optional[str]=None,
                    ) -> Union[dict, None]:
        """Creates a stop market order.

        :param symbol: name of symbol
        :type symbol: str
        :param side: 'SELL' or 'BUY'
        :type side: str
        :param base_qty: bae asst quantitiy
        :type base_qty: str
        :param stop_price: the stop (trigger) price
        :type stop_price: str
        :param client_order_id: arbitrary client order id, defaults to None
        :type client_order_id: Optional[str], optional
        :return: a dictionary with the order id and (if applicable) 
        borrow information 
        :rtype: Union[dict, None]
        """
        pass
    
    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, list]:
        """Cancels an order

        :param order_id: the order id to cancel
        :type order_id: str
        :return: {
            "cancelledOrderIds": [
            "5bd6e9286d99522a52e458de" //orderId
            ]
        }
        :rtype: Dict[str, list]
        """
        pass
    
    @abc.abstractmethod
    def cancel_all_orders(self, symbol: Optional[str]=None) -> Union[dict, None]:
        """Cancels all active orders (for one or all symbols).

        :param symbol: name of symbol, defaults to None
        :type symbol: Optional[str], optional
        :return: {
            "cancelledOrderIds": [
                "5c52e11203aa677f33e493fb", //orderId "5c52e12103aa677f33e493fe", "5c52e12a03aa677f33e49401", "5c52e1be03aa677f33e49404", "5c52e21003aa677f33e49407", "5c6243cb03aa67580f20bf2f", "5c62443703aa67580f20bf32", "5c6265c503aa676fee84129c", "5c6269e503aa676fee84129f", "5c626b0803aa676fee8412a2"
                ]
            }
        :rtype: Union[dict, None]    
        """
        pass
    

# =============================================================================
class IExchangeMargin(abc.ABC):

    @abc.abstractmethod
    def get_account(self):
        pass    
    
    @abc.abstractmethod
    def get_fees(self) -> list:
        pass
        
    # .........................................................................
    @abc.abstractmethod
    def get_margin_risk_limit(self) -> Union[dict, None]:
        """Gets the margin risk/borrow limits.

        :return: [
            {
                "currency": "BTC",
                "borrowMaxAmount": "50",
                "buyMaxAmount": "50",
                "precision": 8 }, ...
            }
        ]
        :rtype: Union[dict, None]
        """
        pass

    @abc.abstractmethod
    def get_margin_config(self) -> dict:
        pass
    
    @abc.abstractmethod
    async def get_borrow_details(self, asset: Optional[str]=None
                                 ) -> Union[Tuple[dict], None]:
        """_summary_

        :param asset: name of the specific asset/coin
        :type asset: str
        :return: {
                "availableBalance": "990.11",
                "currency": "USDT",
                "holdBalance": "7.22",
                "liability": "66.66",
                "maxBorrowSize": "88.88",
                "totalBalance": "997.33", 
            }
        :rtype: Union[Tuple[dict], None]
        """
        pass
    
    @abc.abstractmethod
    def get_liability(self, asset: Optional[str]) -> Union[list, None]:
        """Gets the liability information for one ore all assets.

        :param asset: name 
        :type asset: str
        :return: [
            {
                "accruedInterest": "0.22121",
                "createdAt": "1544657947759",
                "currency": "USDT",
                "dailyIntRate": "0.0021",
                "liability": "1.32121",
                "maturityTime": "1544657947759",
                "principal": "1.22121",
                "repaidSize": "0",
                "term": 7,
                "tradeId": "1231141"
            }
        ]
        :rtype: Union[list, None]
        """
        pass
    
    @abc.abstractmethod
    def borrow(self, currency:str, size: float, type: str='FOK', 
               max_rate: Optional[float]=None, 
               term: Optional[str]=None) -> dict:
        """Borrows the specified asset/currency.

        :param currency: name of the asset (e.g. 'BTC')
        :type currency: str
        :param size: how much to borrow
        :type size: float
        :param type: just set this to FOK, defaults to 'FOK'
        :type type: str, optional
        :param max_rate: cap on daily interest rate, defaults to None
        :type max_rate: Optional[float], optional
        :param term: '7d'/'14d', '28d', defaults to None
        :type term: Optional[str], optional
        :return: {
            "orderId": "a2111213",
            "currency": "USDT"
        }
        :rtype: dict
        """
        pass
    
    @abc.abstractmethod
    def repay(self,  currency: str, size: float, 
              trade_id: Optional[str]=None, 
              sequence: Optional[str]='HIGHEST_RATE_FIRST'
              ) -> Union[dict, None]:
        """Repays a loan.

        :param currency: name of the currency/asset
        :type currency: str
        :param size: how much to repay
        :type size: float
        :param trade_id: a trade id - not necessary, defaults to None
        :type trade_id: Optional[str], optional
        :param sequence: repay sequence, defaults to 'HIGHEST_RATE_FIRST'
        :type sequence: Optional[str], optional
        :return: _description_
        :rtype: Union[dict, None]
        """
        pass

    

        
    
