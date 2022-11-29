"""Home of the code for time-series analysis of temperature data."""

from os import listdir
from os.path import isfile, join

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA


def load_user_data(user_id, data_dir):
    """return pd.DataFrame from joined user's data from multiple files for same user."""
    fn_prefix = f"USER_{user_id:03}_"
    filenames = sorted(
        [
            filename
            for filename in listdir(data_dir)
            if isfile(join(data_dir, filename))
            and (filename.endswith("html") or filename.endswith("xlsx"))
            and fn_prefix in filename
        ],
        key=lambda x: x.split(".")[-2][-2:],  # sort by _DX.html (the day info in name)
    )
    dataframes = [
        pd.read_csv(join(data_dir, fn))
        if fn.endswith("html")
        else pd.read_excel(join(data_dir, fn))
        for fn in filenames
    ]
    df = pd.concat(dataframes)
    df = df[["Time", "Temperature (F)"]]
    df["Time"] = pd.to_datetime(
        df["Time"], unit=("s" if df["Time"].dtype == int else None)
    )
    df = handle_outliers(df, "Temperature (F)")
    df = df[["Time", "Temperature (F)", "Temperature (Outliers Fixed)"]]
    df = df.rename(
        columns={
            "Time": "ds",
            "Temperature (F)": "torig",
            "Temperature (Outliers Fixed)": "y",
        }
    )
    return df


def handle_outliers(df, col_name, threshold=1.5):
    """return pd.DataFrame with replaced outliers with interpolated values."""
    z = np.abs(stats.zscore(df[col_name]))
    ind_outliers = z > threshold
    new_col_name = "Temperature (Outliers Fixed)"
    df[new_col_name] = df[col_name].mask(ind_outliers).interpolate()
    df.reindex()
    return df


def load_user_day_data(filename, data_dir):
    """return pd.DataFrame with user data for a particular day (file)."""
    df = pd.read_csv(join(data_dir, filename))
    df = handle_outliers(df, "Temperature (F)")
    df["Time"] = pd.to_datetime(df["Time"])
    df = df[["Time", "Temperature (F)", "Temperature (Outliers Fixed)"]]
    df = df.rename(
        columns={
            "Time": "ds",
            "Temperature (F)": "torig",
            "Temperature (Outliers Fixed)": "y",
        }
    )
    return df


def draw_results(df):
    """present results in graphical form."""
    fig, axs = plt.subplots(2, 1, layout="constrained")

    # Draw Temperature Series
    t = df["ds"]
    axs[0].plot(t, df["torig"], t, df["y"])
    axs[0].set_xlabel("Time")
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs[0].set_ylabel("Temperature [F]")

    # Draw Outliers
    sns.boxplot(df["torig"], ax=axs[1])
    axs[1].set_xlabel("Detecting Outliers")
    axs[1].set_ylabel("Temperature [F]")

    fig.savefig("outliers.png")


def train_and_predict_prophet_model(df):
    """train prophet model fitted on dataframe and draw a graph."""
    df = df[["ds", "y"]]
    predlen = 30  # Use this setting for 2-min frequency
    # predlen = 3600  # Use this setting for 1-sec frequency
    times_look_back = 10  # how many times * predlen to draw on a graph
    trainlen = len(df) - predlen
    m = Prophet(changepoint_prior_scale=0.5).fit(df.iloc[:trainlen])
    fcst = m.predict(df)
    fig = m.plot(fcst, include_legend=True)
    ax = fig.gca()
    fig.savefig("complete.png")
    ax.set_xlim(df["ds"].iloc[-times_look_back * predlen], df["ds"].iloc[-1])
    ax.plot(df["ds"].iloc[-predlen:], df["y"].iloc[-predlen:], "r.")
    fig.savefig("forecast.png")


def train_and_predict_arima(df):
    """create ARIMA model and save graph with predictions."""
    predlen = 30
    trainlen = len(df) - predlen
    t = df["ds"]
    y = df["y"]

    # Fit
    y_train = y[:trainlen]
    am = ARIMA(y_train, order=(10, 2, 40)).fit()
    plt.plot(t[:trainlen], y[:trainlen], "b", label="train")
    plt.plot(t[trainlen:], y[trainlen:], "r.", label="recorded")

    # Forecast
    times_look_back = 5
    forecastlen = len(df) - trainlen
    preds = am.forecast(forecastlen)
    plt.plot(t[-predlen:], preds, "c", label="forecast")
    ax = plt.gca()
    ax.set_xlabel("Time")
    ax.set_xlabel("Temperature [F]")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(df["ds"].iloc[-times_look_back * predlen], df["ds"].iloc[-1])
    plt.savefig("arima.png")


if __name__ == "__main__":
    data_dir = "./data/"
    # user_day_file = "USER_001_022E1E_D1.html"
    # user_day_file = "USER_002_0232ED_D1.html"
    # df = load_user_day_data(user_day_file, data_dir)
    user_id = 1
    df = load_user_data(user_id, data_dir)
    # draw_results(df)
    train_and_predict_prophet_model(df)
    # train_and_predict_arima(df)
