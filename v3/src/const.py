import datetime as dt

dataPath = 'data/'
plotPath = 'plot/'
vn30Path = 'src/'

start_date = '4/12/2014'
# end = '01/01/2020'
end_date = dt.datetime.now().strftime('%d/%m/%Y')
country = 'VietNam'


def load_vn30():
    with open(f'{vn30Path}vn30.txt') as f:
        vn30 = f.read().splitlines()
    return vn30


symbol = 'HPG'

real_price_str = 'Real Price'
prediction_price_str = 'Prediction Price'
price_str = 'Price'
time_str = 'Time'
