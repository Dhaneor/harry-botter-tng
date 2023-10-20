#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 17:45:23 2022

@author_ dhaneor
"""
import re
from datetime import datetime, timedelta
from typing import Union, List

from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.orm import relationship, joinedload, validates
from sqlalchemy import Column, ColumnDefault, ForeignKey
from sqlalchemy import Integer, Float, String, Boolean, DateTime
from sqlalchemy import create_engine, inspect, select, delete

from broker.config import DB_USER, DB_PASSWORD, DB_HOST
from broker.config import (VALID_EXCHANGES, VALID_MARKETS, MAX_RISK_LEVEL,
                    MAX_USER_LEVERAGE, MIN_INITIAL_CAPITAL,
                    MAX_INITIAL_CAPITAL)


engine = create_engine(
    f'mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:3306/akasha'
    )
Base = declarative_base()


# =============================================================================
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False, unique=True)
    password = Column(String(128), nullable=False)
    email = Column(String(128), nullable=False, unique=True)
    telegram = Column(String(128))
    vip_level = Column(Integer, nullable=False, default=1)
    date_registered = Column(DateTime, nullable=False)
    expires = Column(DateTime, nullable=False)
    is_active = Column(Boolean, nullable=False, default=False)

    accounts = relationship('Account', back_populates='users')

    def __repr__(self):
        return f"User(id={self.id!r}, name={self.name!r}, email={self.email!r})"

    @validates('name')
    def validate_username(self, key, username):
        if not username:
            raise AssertionError('No username provided')
        if len(username) < 5 or len(username) > 20:
            raise AssertionError(
                'Username must be between 5 and 20 characters'
            )
        return username

    @validates('password')
    def validate_password(self, key, password):
        if not password:
            raise AssertionError('No password provided')
        if len(password) < 5 or len(password) > 128:
            raise AssertionError(
                'Password must be between 5 and 20 characters'
            )
        return password

    @validates('email')
    def validate_email(self, key, email):
        if not email:
            raise AssertionError('No email provided')
        if not re.match("[^@]+@[^@]+\.[^@]+", email):
            raise AssertionError('Provided email is not valid')
        return email

    @validates('telegram')
    def validate_telegram(self, key, telegram):
        if telegram:
            if not re.match("@+[^@]", telegram):
                raise AssertionError('Provided Telegram name is not valid')
        return telegram

    @validates('vip_level')
    def validate_vip_level(self, key, vip_level):
        if not 1 <= vip_level <= 11:
            raise AssertionError('VIP level must be between 1 and 10')


class Account(Base):
    __tablename__ = 'exchange_accounts'

    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)
    is_active = Column(Boolean(), default=False)
    api_key = Column(String(128), nullable=False, unique=True)
    api_secret = Column(String(128), unique=True)
    api_passphrase = Column(String(128))
    date_created = Column(DateTime(), nullable=False)
    exchange = Column(String(32), nullable=False)
    market = Column(String(32), nullable=False)
    assets = Column(String(512))
    quote_asset = Column(String(16), nullable=False)
    interval = Column(String(8), nullable=False)
    strategy = Column(String(128), nullable=False)
    risk_level = Column(Integer, nullable=False)
    max_leverage = Column(Float(), ColumnDefault(3), nullable=False)
    initial_capital = Column(Float())

    user_id = Column(Integer,  ForeignKey("users.id"), nullable=False)
    users = relationship('User', back_populates='accounts')

    def __repr__(self):
        str_ = f"Account([id={self.id}, name={self.name}, "\
               f"exchange={self.exchange}, " \
               f"date_created={self.date_created})"

        return  str_

    @validates('exchange')
    def validate_exchange(self, key, exchange):
        if not exchange:
            raise AssertionError('No exchange provided')
        if not isinstance(exchange, str):
            raise AssertionError(
                f'exchange must be a string but was {type(exchange)}'
            )
        if not exchange in VALID_EXCHANGES:
            raise AssertionError(f'{exchange} is not a valid exchange')

        return exchange

    @validates('market')
    def validate_market(self, key, market):
        if not market:
            raise AssertionError('No market provided')
        if not isinstance(market, str):
            raise AssertionError(
                f'market must be a string but was {type(market)}'
            )
        if not market in VALID_MARKETS:
            raise AssertionError(f'{market} is not a valid exchange')

        return market

    @validates('risk_level')
    def validate_risk_level(self, key, risk_level):
        if not risk_level:
            raise AssertionError('No risk level provided')
        if not isinstance(risk_level, int):
            raise AssertionError(
                f'risk level must be an integer but was {type(risk_level)}'
                )
        if not 1 <= risk_level <= MAX_RISK_LEVEL:
            raise AssertionError(
                f'risk level must be between 1 and {MAX_RISK_LEVEL}'
            )

        return risk_level

    @validates('max_leverage')
    def validate_max_leverage(self, key, max_leverage):
        if not max_leverage:
            raise AssertionError('No value for maximum leverage provided')
        if not isinstance(max_leverage, (float, int)):
            raise AssertionError(
                f'maximum leverage must be integer or float but '\
                    f'was {type(max_leverage)}'
                )
        if not 0.1 <= max_leverage <= MAX_USER_LEVERAGE:
            raise AssertionError(
                f'maximum leverage must be between 0.1 '\
                    f'and {MAX_USER_LEVERAGE}'
            )

        return round(max_leverage, 2)

    @validates('initial_capital')
    def validate_initial_capital(self, key, initial_capital):
        if not isinstance(initial_capital, (float, int)):
            raise AssertionError(
                f'initial capital must be integer or float but '\
                    f'was {type(initial_capital)}'
                )
        if not MIN_INITIAL_CAPITAL <= initial_capital <= MAX_INITIAL_CAPITAL:
            raise AssertionError(
                f'initial capital must be between {MIN_INITIAL_CAPITAL} '\
                    f'and {MAX_INITIAL_CAPITAL}'
            )

        return initial_capital


# =============================================================================
class Accounts:

    def __init__(self, session):
        self.session = session
        self._create_table()

    # -------------------------------------------------------------------------
    @property
    def all_accounts(self):
        with self.session() as session:
            all_accounts = session.query(Account).all()

        return all_accounts

    # -------------------------------------------------------------------------
    def create_account(self, user_id: int, account_params:dict):

        return Account(
            user_id=user_id,
            name=account_params.get('name'),
            is_active=account_params.get('is active', True),
            api_key=account_params.get('api_key'),
            api_secret=account_params.get('api_secret'),
            api_passphrase=account_params.get('api_passphrase'),
            date_created=datetime.now(),
            exchange=account_params.get('exchange'),
            market=account_params.get('market'),
            assets=account_params.get('assets'),
            quote_asset=account_params.get('quote asset'),
            interval=account_params.get('interval'),
            strategy=account_params.get('strategy'),
            risk_level=account_params.get('risk level', 1),
            max_leverage=account_params.get('max leverage', 1),
            initial_capital=account_params.get('initial capital'),
            )

    def get_account(self, id: int) -> Union[Account, None]:
        with self.session() as session:
            res = session.query(Account).filter(Account.id==id).all()

        return res[0] if res else None

    def get_accounts(self, user_name:str):

        user = self.users.get_user(user_name)
        print(user)

        return user.accounts

    def update_account(self, id:int, update_values:dict):
        account = self.get_account(id)

        if not account:
            raise ValueError(f'account id {id} does not exist!')

        with self.session() as session:

            for k,v in update_values.items():
                if hasattr(account, k):
                    setattr(account, k, v)
                else:
                    raise ValueError(f'Account has no attribute {k}')

            session.add(account)
            session.commit()

    def delete_account(self, id:int):
        with self.session() as session:
            session.execute(delete(Account).where(Account.id == id))
            session.commit()

    # -------------------------------------------------------------------------
    def _create_table(self):
        table_exists = inspect(engine).has_table("exchange_accounts")

        if not table_exists:
            Base.metadata.create_all(engine)


class Users:

    def __init__(self):
        self.session = sessionmaker()
        self.session.configure(bind=engine)

        self.accounts: Accounts = Accounts(self.session)

    # -------------------------------------------------------------------------
    @property
    def all_user_names(self) -> List[str]:
        with self.session() as session:
            all_users = session.query(User).all()

        return [u.name for u in all_users]

    @property
    def all_users(self) -> List[User]:
        with self.session() as session:
            res = session.query(User)\
                .options(joinedload(User.accounts))\
                    .all()

        return res

    # -------------------------------------------------------------------------
    def create_user(self, user_info: dict):
        # self._create_tables()

        with self.session() as session:

            username = user_info.get('user_name')
            if session.\
                query(session.query(User).filter(User.name == username).exists())\
                    .scalar():
                raise AssertionError('Username is already in use')

            email = user_info.get('email')
            if session.\
                query(session.query(User).filter(User.email == email).exists())\
                    .scalar():
                raise AssertionError('This email address is already registered')

            new_user = User(
                name = username,
                password = user_info.get('password'),
                email = email,
                telegram = user_info.get('telegram', None),
                vip_level = user_info.get('vip_level', 1),
                date_registered = datetime.now(),
                expires = datetime.now() + timedelta(days=365)
            )

            session.add(new_user)
            session.commit()

    def get_user(self, user_name: str) -> Union[User, None]:
        with self.session() as session:
            res = session.query(User)\
                .options(joinedload(User.accounts))\
                    .filter(User.name==user_name).all()

        return res[0] if res else None

    def get_user_by_id(self, user_id: int) -> Union[User, None]:
        with self.session() as session:
            res = session.query(User)\
                .options(joinedload(User.accounts))\
                    .filter(User.id==user_id).all()

        return res[0] if res else None

    def update_user(self, user_name:str, update_values:dict):
        with self.session() as session:
            user = session.execute(
                select(User).filter_by(name=user_name)
                ).scalar_one()

            for k,v in update_values.items():
                if hasattr(user, k):
                    setattr(user, k, v)

            session.commit()

    def delete_user(self, user_name):
        with self.session() as session:
            session.execute(delete(User).where(User.name == user_name))
            session.commit()

    # -------------------------------------------------------------------------
    def add_account_to_user(self, user_id: int, account_params: dict):
        user = self.get_user_by_id(user_id)

        if not user:
            raise ValueError('no user found with id: {user_id}')

        if user.accounts:
            for acc in user.accounts:
                try:
                    if acc.api_key == account_params['api_key']:
                        raise AssertionError(
                            f'the provided api key is already assigned to one'\
                                f'of the users accounts. please provide '\
                                    f'another api key/secret/passphrase!'
                        )
                except Exception as e:
                    raise AssertionError(e)

        with self.session() as session:
            account = self.accounts.create_account(
                user_id=user_id, account_params=account_params
            )
            session.add(user)
            session.add(account)
            session.commit()

            return account.id

        return None

    def get_accounts_for_user(self, user_id: int,
                              account_id: Union[int, None]=None
                              ) -> Union[List[Account], None]:
        user = self.get_user_by_id(user_id)

        if not user:
            raise ValueError(f'user with id {user_id} does not exist')

        if not account_id:
            return user.accounts
        else:
            if user.accounts:
                return [acc for acc in user.accounts if acc.id == account_id]
            else:
                return None

    def get_account_by_id(self, account_id: int) -> Union[Account, None]:
        return self.accounts.get_account(id=account_id)

    def update_account(self, account_id: int, update_values: dict):
        self.accounts.update_account(id=account_id, update_values=update_values)

    def delete_account(self, account_id: int):
        try:
            self.accounts.delete_account(account_id)
        except Exception as e:
            raise ValueError(f'account with id {account_id} does not exist')

    # -------------------------------------------------------------------------
    def _create_tables(self):
        table_exists = inspect(engine).has_table("users")

        if not table_exists:
            Base.metadata.create_all(engine)











"""
from SQLAlchemy Tutorial ... setting up of two interlinked objects/tables:

>>> class User(Base):
...     __tablename__ = "user_account"
...
...     id = Column(Integer, primary_key=True)
...     name = Column(String(30))
...     fullname = Column(String)
...
...     addresses = relationship(
...         "Address", back_populates="user", cascade="all, delete-orphan"
...     )
...
...     def __repr__(self):
...         return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"

>>> class Address(Base):
...     __tablename__ = "address"
...
...     id = Column(Integer, primary_key=True)
...     email_address = Column(String, nullable=False)
...     user_id = Column(Integer, ForeignKey("user_account.id"), nullable=False)
...
...     user = relationship("User", back_populates="addresses")
...
...     def __repr__(self):
...         return f"Address(id={self.id!r}, email_address={self.email_address!r})"
"""
