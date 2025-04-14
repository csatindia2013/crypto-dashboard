import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import openai
from datetime import datetime
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Securely pull from environment
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
REFRESH_INTERVAL = 60

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp - tp.rolling(period).mean()) / (0.015 * tp.rolling(period).std())

def compute_sma(series, window): return series.rolling(window).mean()
def compute_ema(series, window): return series.ewm(span=window, adjust=False).mean()
def compute_bollinger_bands(series, window=20):
    sma = compute_sma(series, window)
    std = series.rolling(window).std()
    return sma + 2*std, sma - 2*std

def compute_stochrsi(series, period=14):
    rsi = compute_rsi(series, period)
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    return 100 * (rsi - min_rsi) / (max_rsi - min_rsi)

app.layout = dbc.Container([
    html.H1("📊 Crypto Dashboard (Render Deploy)", className="text-center my-4"),
    dbc.Row([
        dbc.Col(dcc.Input(id='ticker-input', value='BTC-USD', type='text', className='form-control'), md=6),
        dbc.Col(html.Div([
            html.Button('Refresh Now', id='submit-button', className='btn btn-primary w-100 mb-2'),
            dcc.Interval(id='auto-refresh', interval=60*1000, n_intervals=0),
            dcc.Interval(id='countdown-timer', interval=1000, n_intervals=0),
        ]), md=2),
    ], className="mb-3"),
    dbc.Row(dbc.Col(html.Div(id='countdown-display'))),
    dbc.Row(dbc.Col(html.Div(id='price-display'))),
    dbc.Row(dbc.Col(html.Div(id='indicator-values'))),
    dbc.Row(dbc.Col(html.Div(id='chatgpt-commentary'))),
    dbc.Row(dbc.Col(dcc.Graph(id='price-chart')))
], fluid=True)

@app.callback(
    Output('countdown-display', 'children'),
    Input('auto-refresh', 'n_intervals'),
    Input('countdown-timer', 'n_intervals')
)
def update_countdown(_, tick):
    seconds = 60 - (tick % 60)
    return f"🔄 Auto-refreshing in {seconds} sec"

@app.callback(
    [Output('price-display', 'children'),
     Output('indicator-values', 'children'),
     Output('chatgpt-commentary', 'children'),
     Output('price-chart', 'figure')],
    [Input('submit-button', 'n_clicks'),
     Input('auto-refresh', 'n_intervals')],
    [State('ticker-input', 'value')]
)
def update_dashboard(_, __, ticker):
    try:
        df = yf.download(ticker, period='6mo', interval='1d')
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
        df['CCI'] = compute_cci(df)
        df['SMA_20'] = compute_sma(df['Close'], 20)
        df['SMA_50'] = compute_sma(df['Close'], 50)
        df['EMA_12'] = compute_ema(df['Close'], 12)
        df['EMA_26'] = compute_ema(df['Close'], 26)
        df['BB_upper'], df['BB_lower'] = compute_bollinger_bands(df['Close'])
        df['StochRSI'] = compute_stochrsi(df['Close'])
        latest = df.dropna().iloc[[-1]]
        ltt = latest.index[0].strftime('%b %d, %Y')
        price = float(latest['Close'].iloc[0])
        rsi = float(latest['RSI'].iloc[0])
        macd = float(latest['MACD'].iloc[0])
        signal = float(latest['MACD_Signal'].iloc[0])
        cci = float(latest['CCI'].iloc[0])
        summary = {"RSI": round(rsi, 2), "MACD": round(macd, 4), "MACD_Signal": round(signal, 4), "CCI": round(cci, 2)}
        prompt = f"Based on these indicators, give a one-word recommendation (Buy/Sell/Hold) with one-line reason:\n\n{summary}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=60
        )
        gpt_text = response.choices[0].message.content.strip()
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', marker_color='orange'), row=2, col=1)
        fig.update_layout(height=600, margin=dict(t=40, b=40), legend_title_text='Indicators')
        return (
            f"{ticker.upper()} • ${round(price, 2)} • LTT: {ltt}",
            f"RSI: {rsi} | MACD: {macd} | Signal: {signal} | CCI: {cci}",
            gpt_text, fig
        )
    except Exception as e:
        return str(e), "", "", go.Figure()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
