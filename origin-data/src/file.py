def export_csv(df, file_name):
    df.to_csv(f'{file_name}.csv', index=False)


def export_json(df, file_name):
    df.to_json(f'{file_name}.json', index=False)


def export_png(plt, file_name):
    plt.savefig(f'{file_name}.png')