import ta
import pandas as pd
from exchange.cBot_binance_futur  import *
from time import sleep, time
from strategies.HarmonicDivergences.HarmonicDivergences import *
from config.BotConfig import *
from datetime import datetime

exchange= {}
exchange = cBot_binance_futur(
        apiKey = api_key,
        secret = api_secret,
    )

# -- Strategy variable --

perpSymbol = ticker + stake
strategie = HarmonicDivergences(exchange)

try:
    strategie.backsteps(perpSymbol, timeframe, perDay * nbDays)

except BaseException as err:
    print("Unexpected", err)
    pass

