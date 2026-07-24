import plotly.express as px


def temperature_chart(df):

    fig = px.line(
        df,
        x="date",
        y="temperature",
        title="Temperature Trend"
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    return fig


def humidity_chart(df):

    fig = px.line(
        df,
        x="date",
        y="humidity",
        title="Humidity Trend"
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    return fig


def rainfall_chart(df):

    fig = px.bar(
        df,
        x="date",
        y="rainfall",
        title="Rainfall Trend"
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    return fig


def wind_chart(df):

    fig = px.line(
        df,
        x="date",
        y="wind_speed",
        title="Wind Speed Trend"
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    return fig