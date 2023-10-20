#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 23:38:53 2022

@author: dhaneor
"""
import asyncio
import aiohttp
import json
import hmac
import hashlib
import base64
import time
from uuid import uuid1
from urllib.parse import urljoin
from pprint import pprint

from broker.config import CREDENTIALS

class AsyncKucoinBaseRestApi(object):

    def __init__(self, key='', secret='', passphrase='', is_sandbox=False, url='', is_v1api=False):
        """
        https://docs.kucoin.com

        :param key: Api Token Id  (Mandatory)
        :type key: string
        :param secret: Api Secret  (Mandatory)
        :type secret: string
        :param passphrase: Api Passphrase used to create API  (Mandatory)
        :type passphrase: string
        :param is_sandbox: True sandbox , False  (optional)
        """

        if url:
            self.url = url
        else:
            if is_sandbox:
                self.url = 'https://openapi-sandbox.kucoin.com'
            else:
                self.url = 'https://api.kucoin.com'

        self.key = key
        self.secret = secret
        self.passphrase = passphrase
        self.is_v1api = False # is_v1api

    async def _request(self, method, uri, timeout=5, auth=True, params=None):
        uri_path = uri
        data_json = ''
        if method in ['GET', 'DELETE']:
            if params:
                strl = []
                for key in sorted(params):
                    strl.append("{}={}".format(key, params[key]))
                data_json += '&'.join(strl)
                uri += '?' + data_json
                uri_path = uri
        else:
            if params:
                data_json = json.dumps(params)

                uri_path = uri + data_json

        headers = {}
        if auth:
            now_time = int(time.time()) * 1000
            str_to_sign = str(now_time) + method + uri_path
            sign = base64.b64encode(
                hmac.new(
                    self.secret.encode('utf-8'),
                    str_to_sign.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('ascii')
            if self.is_v1api:
                headers = {
                    "KC-API-SIGN": sign,
                    "KC-API-TIMESTAMP": str(now_time),
                    "KC-API-KEY": self.key,
                    "KC-API-PASSPHRASE": self.passphrase,
                    "Content-Type": "application/json"
                }
            else:
                passphrase = base64.b64encode(
                    hmac.new(
                        self.secret.encode('utf-8'),
                        self.passphrase.encode('utf-8'),
                        hashlib.sha256
                    ).digest()
                ).decode('ascii')
                headers = {
                    "KC-API-SIGN": sign,
                    "KC-API-TIMESTAMP": str(now_time),
                    "KC-API-KEY": self.key,
                    "KC-API-PASSPHRASE": passphrase,
                    "Content-Type": "application/json",
                    "KC-API-KEY-VERSION": "2"
                }
        headers["User-Agent"] = "kucoin-python-sdk/v1.0.0"
        url = urljoin(self.url, uri)

        if method == 'GET':
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url=url) as r:
                    response = await r.json() # .read()
        else:
            raise NotImplementedError(
                f'Kucoin Async Client: method {method} not implemented'
            )

        # else:
        #     # response_data = requests.request(method, url, headers=headers, data=data_json,
        #     #                                  timeout=timeout)
        #     async with aiohttp.ClientSession(
        #         headers=headers, timeout=timeout
        #     ) as session:
        #         async with session.post(url, data=data_json) as r:
        #             response_data = await r.json()

        return self.check_response_data(response)

    @staticmethod
    def check_response_data(response_data):
        try:
            if '200' in response_data['code']:
                return response_data['data']
        except ValueError:
            raise Exception(f'invalid data format: {response_data}')

        code, msg = response_data['code'], response_data['message']
        raise Exception(f'{code} - {msg}')


    @property
    async def return_unique_id(self):
        return ''.join([each for each in str(uuid1()).split('-')])


class AsyncKucoinClient(AsyncKucoinBaseRestApi):

    def __init__(self, credentials: dict, is_sandbox: bool=False):
        AsyncKucoinBaseRestApi.__init__(
            self,
            key=credentials['api_key'],
            secret=credentials['api_secret'],
            passphrase=credentials['api_passphrase'],
            is_sandbox=is_sandbox
        )

    async def get_margin_account(self):
        params = {'type': 'margin'}
        res = await self._request(
            method='GET',
            uri='/api/v1/margin/account',
            auth=True,
            params=params
        )
        return res['accounts']

# =============================================================================
async def main():

    c = AsyncKucoinClient(credentials=CREDENTIALS)

    res = await asyncio.gather(*[c.get_margin_account() for i in range(3)])
    data = res[-1]

    [print(i) for i in data]
    print(f'got {len(data)} balances')

    await asyncio.sleep(1)


# =============================================================================
#                                   MAIN                                      #
# =============================================================================
if __name__ == '__main__':
    asyncio.run(main())