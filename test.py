from vnstock import Vnstock
from data_utils import clean_data

stock = Vnstock().stock(symbol="VIB")

df = stock.quote.history(
    start="2023-01-01",
    end="2026-04-07",
    interval="1D"
)

df = clean_data(df)
print(df)

