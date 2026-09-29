import yfinance as yf

spx = yf.Ticker("^GSPC").history(start="2010-01-01")
close = spx["Close"]
close.index = close.index.tz_localize(None) # drops timezone
close.to_csv("spx_close.csv")