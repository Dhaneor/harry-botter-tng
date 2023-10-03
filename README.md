Harry Botter is a crypto trading system. As of now it works on Kucoin and in the Cross Margin market.

FEATURES:

- leveraged trading in the Cross Margin market in any timeframe that is provided by the exchange
- dynamic position sizing based on risk and use of a risk target based on volatility
- multi-user/multi-account capability
- multiple symbols per account with automatic risk management and improved risk profile while also using a diversification multiplier for position sizes
- state-less operation so it doesn't matter if the program or the server is restarted
- stop loss and take profit orders

It's still in the Beta stage, but it is working. You just have to know what to do to make it so. There are many components that still need improvement and hardening.

TODO:

- build a web app that allows for convenient monitoring of the operation
- enhance the Users module, so we can use the database for information about the
users, exchange accounts and related strategies/markets to trade in
- restructure the Oracle (strategies) module, so we are more flexible when building strategies
- drop the outdated Kucoin library and replace it with CCXT. This enables more exchanges
- provide a Docker version for easy installation 
