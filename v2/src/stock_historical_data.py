import investpy as ip
import pandas as pd
import const
import file


class StockHistoricalData:
    def __init__(self, df=None, symbol=const.symbol, country=const.country, start_date=const.start_date,
                 end_date=const.end_date):
        self.df = df
        self.symbol = symbol
        self.country = country
        self.start_date = start_date
        self.end_date = end_date

    def get_data(self):
        shd = ip.get_stock_historical_data(
            stock=self.symbol, country=self.country, from_date=self.start_date, to_date=self.end_date)
        df = pd.DataFrame(shd)

        # df[const.oc_str] = df[const.open_str] - df[const.close_str]
        # df[const.hl_str] = df[const.high_str] - df[const.low_str]
        #
        # df[f'{const.sma}{const.ma1}'] = df[const.close_str].rolling(window=const.ma1).mean()
        # df[f'{const.sma}{const.ma2}'] = df[const.close_str].rolling(window=const.ma2).mean()
        # df[f'{const.sma}{const.ma3}'] = df[const.close_str].rolling(window=const.ma3).mean()
        # df.dropna(inplace=False)

        self.df = df

    def export_file(self):
        file.export_csv(self.df, f'{const.dataPath}{const.symbol}')
