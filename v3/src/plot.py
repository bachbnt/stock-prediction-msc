import matplotlib.pyplot as plt
import const
import file


def draw(real_price, prediction_price, real_color='red', prediction_color='blue', real_label=const.real_price_str,
         prediction_label=const.prediction_price_str):
    plt.plot(real_price, color=real_color, label=real_label)
    plt.plot(prediction_price, color=prediction_color, label=prediction_label)
    plt.title(f'{const.symbol}')
    plt.xlabel(const.time_str)
    plt.ylabel(const.price_str)
    plt.ylim(bottom=0)
    plt.legend()
    file.export_png(plt, f'{const.plotPath}{const.symbol}')
    # plt.show()
