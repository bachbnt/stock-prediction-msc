import datetime as dt

dataPath = 'data/'
plotPath = 'plot/'
vn30Path = 'src/'

start_date = '4/12/2014'
# end = '01/01/2020'
end_date = dt.datetime.now().strftime('%d/%m/%Y')
country = 'VietNam'

with open(f'{vn30Path}vn30.txt') as f:
    vn30 = f.read().splitlines()
symbol = 'ACB'

ma1 = 7
ma2 = 14
ma3 = 21
open_str = 'Open'
close_str = 'Close'
high_str = 'High'
low_str = 'Low'
oc_str = 'O-C'
hl_str = 'H-L'
sma = 'SMA'

real_price_str = 'Real Price'
prediction_price_str = 'Prediction Price'
price_str = 'Price'
time_str = 'Time'
