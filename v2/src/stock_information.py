import investpy as ip
import pandas as pd
import const
import file


class StockInformation:
    def __init__(self, df=None, symbol=const.symbol, country=const.country):
        self.df = df
        self.symbol = symbol
        self.country = country

    def get_data(self):
        si = ip.get_stock_information(stock=self.symbol, country=self.country)
        df = pd.DataFrame(si)

        self.df = df

    def export_file(self):
        file.export_csv(self.df, f'{const.dataPath}{const.symbol}i')
