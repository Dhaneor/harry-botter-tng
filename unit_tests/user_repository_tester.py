#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""

from datetime import datetime, timedelta
from random import choice, random,randint
from pydantic.error_wrappers import ValidationError
from string import ascii_letters, digits
from pprint import pprint

# -----------------------------------------------------------------------------
# make sure all imports from parent directory work
import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)
# -----------------------------------------------------------------------------

from models.users import Users
from analysis.oracle import Oracle
from util.timeops import execution_time
from  broker.config import CREDENTIALS, VALID_EXCHANGES, VALID_MARKETS

USERS = Users()

NAMES = ['fred', 'kevin', 'sarah', 'max', 'catherine', 'jenny', 'arthur',
         'jonathan', 'trevor', 'juan', 'emilie', 'chen', 'karla']

INTERVALS = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d']

ASSETS = ['BTC', 'ETH', 'ADA', 'XRP', 'DOGE', 'ATOM', 'AVAX', 'BNB']

ORACLE = Oracle()

# =============================================================================
def create_test_user():

    user_name = 'dhaneor'

    user_params = {
        'user_name' : user_name,
        'password' : 123456,
        'email' : 'dhaneor@cloud7.org',
        'vip_level' : 10,
    }

    try:
        USERS.create_user(user_params)
    except ValidationError as e:
        print(f'Validation error for : {user_params}')
        # print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)

    # .........................................................................
    account_params = {
        'name' : 'dhaneor_developer',
        'exchange' : 'kucoin',
        'api_key' : CREDENTIALS['api_key'],
        'api_secret' : CREDENTIALS['api_secret'],
        'api_passphrase' : CREDENTIALS['api_passphrase'],
    }

    ACCOUNTS.create_account(user_name, account_params)

    # .........................................................................
    # print(USERS.get_user('dhaneor'))
    pprint(ACCOUNTS.get_accounts('dhaneor'))
    print('•'*80 + '\n')


# =============================================================================
def create_random_string(length:int):
    return ''.join([
        choice([*ascii_letters, *digits]) for _ in range(length)
    ])

def get_random_assets():
    assets = list(
        set(
            [choice(ASSETS) for _ in range(randint(1, 10))]
        )
    )

    return ' '.join(assets)

def get_random_account():
    account, counter = None, 0
    while not account and counter < 100:
        user = choice(USERS.all_users)
        if user and user.accounts:
            idx = randint(0, len(user.accounts)-1)
            return user.accounts[idx]
        counter += 1

# =============================================================================
@execution_time
def test_create_user(no_of_new_users=1, verbose=True):

    for _ in range(no_of_new_users):

        name = f'{choice(NAMES)}_{int(random()*1000)}' if random() < 0.75 else None
        email = f'{name}@cloud7.org' if random() < 0.75 else None
        password = '123456' if random() < 0.75 else None

        if verbose:
            print(f'creating user: {name} (email: {email})')

        params = {
            'user_name' : name,
            'password' : password,
            'email' : email,
        }

        try:
            USERS.create_user(params)
        except (AssertionError, ValidationError) as e:
            print(f'Validation error for : {params}')
            print(e)
        except ValueError as e:
            print(f'ValueError: {e}')
        except Exception as e:
            print(f'ValueError: {e}')


@execution_time
def test_get_user(user_name):
    user = USERS.get_user(user_name)

    if not user:
        print(f'unable to find user: {user_name}')
        return

    print(user)
    print('-'*80)
    print(f'user {user_name} has the these accounts configured:')
    if user.accounts:
        [print(acc) for acc in user.accounts]
    else:
        print('None')

@execution_time
def test_update_user(user_name=None, runs=1):
    for _ in range(runs):
        if not user_name:
            user_name = choice(USERS.all_user_names)

        email = f'{create_random_string(5)}_{randint(1, 100)}@wolke7.net'
        email = None if random() < 0.1 else email
        new_values = {'email': email, 'vip level': randint(1, 10)}
        try:
            USERS.update_user(user_name, new_values)
        except AssertionError as e:
            print(e)
        finally:
            print(USERS.get_user(user_name))

@execution_time
def test_delete_user(user_name:str):
    try:
        USERS.delete_user(user_name)
    except Exception as e:
        print(e)

    print(USERS.get_user(user_name))

@execution_time
def test_get_all_user_names():
    print(USERS.all_user_names)

@execution_time
def test_get_all_users():
    for user in USERS.all_users:
        print(user)
        [print('\t', account) for account in user.accounts]


        print('='*120)


# -----------------------------------------------------------------------------
@execution_time
def test_create_account(runs=1):

    assets = get_random_assets()

    for _ in range(runs):
        user = choice(USERS.all_users)

        account_params = {
            'name': f'test_{randint(1, 1000)}',
            'is active': True,
            'api_key': create_random_string(32),
            'api_secret': create_random_string(32),
            'api_passphrase': create_random_string(32),
            'exchange': 'kucoin',
            'market': choice(VALID_MARKETS),
            'assets': assets,
            'quote asset': 'USDT',
            'interval': '1d',
            'strategy': 'Pure Breakout',
            'risk level': randint(1, 6),
            'max leverage': randint(1, 6),
            'initial capital': 1_000
        }

        pprint(account_params)

        try:
            _id = USERS.add_account_to_user(
                user.id, account_params  # type: ignore
            )
        except AssertionError as e:
            print(e)
            _id = None

        if _id:
            print(USERS.get_user_by_id(user.id))

@execution_time
def test_get_account_():
    print(ACCOUNTS.get_account(1))

@execution_time
def test_get_accounts(user_name: str):
    pprint(ACCOUNTS.get_accounts(user_name))

@execution_time
def test_update_account(runs=1):
    for _ in  range(runs):
        account = get_random_account()

        if not account:
            print('unable to get any account ... exiting!')
            return

        update_values = {
            'assets': get_random_assets(),
            'max_leverage': random() * 6,
            'crap': 'useless stuff'
        }

        if random() > 0.1:
            del update_values['crap']

        try:
            USERS.update_account(
                account_id=account.id, update_values=update_values
            )

            user = USERS.get_user_by_id(user.id) # type:ignore
            print(user)
            for acc in user.accounts: # type:ignore
                if acc.id == account.id:
                    pprint(acc.__dict__)

        except (AssertionError, ValueError) as e:
            print(e)

@execution_time
def test_delete_account(runs=1):
    for _ in  range(runs):
        account = get_random_account()
        if not account:
            print('unable to get any account ... giving up now!')
            continue

        try:
            USERS.delete_account(account.id)
            print(f'deleted account id {account.id}')
        except ValueError as e:
            print(f'unable to delete account with id {account.id}: {e}')

        assert USERS.get_account_by_id(account.id) == None

@execution_time
def test_get_all_accounts():
    pprint(ACCOUNTS.all_accounts)

# =========================================================================== #
#                                   MAIN                                      #
# =========================================================================== #
if __name__ == '__main__':
    print(datetime.now().replace(microsecond=0))

    test_create_user(no_of_new_users=50)
    # test_get_user('dhaneor')
    # test_update_user(runs=10)
    # test_delete_user('trevor_65')
    # test_get_all_user_names()
    # test_get_all_users()

    test_create_account(15)
    # test_get_accounts('dhaneor')
    # test_update_account(1)
    test_delete_account(15)
    # test_get_all_accounts()

    # create_test_user()

    # .........................................................................
    # test_create_strategy(10)