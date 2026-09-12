import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ddgs import DDGS
from google import genai


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Phân tích Dầu CLUSDT",
    page_icon="🛢️",
    layout="wide"
)

BINANCE_SYMBOL = "CLUSDT"

# Binance Futures fallback endpoints
BINANCE_FUTURES_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

st.title("🛢️ AI Phân tích Giá Dầu CLUSDT")
st.markdown(
    "Phân tích WTI bằng nến Binance, RSI, MACD, EMA, Volume, "
    "Funding, Open Interest, tin tức và Gemini AI."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ CẤU HÌNH")

api_key = st.sidebar.text_input(
    "🔑 Gemini API Key",
    type="password",
    placeholder="Dán Gemini API Key"
)

st.sidebar.markdown(
    "[👉 Lấy Gemini API Key]"
    "(https://aistudio.google.com/apikey)"
)

interval = st.sidebar.selectbox(
    "⏱️ Khung thời gian",
    ["5m", "15m", "30m", "1h", "4h", "1d"],
    index=3
)

limit = st.sidebar.selectbox(
    "📊 Số nến",
    [100, 200, 300, 500],
    index=1
)

st.sidebar.markdown("---")

st.sidebar.warning(
    "Dữ liệu Binance có thể bị giới hạn theo khu vực/IP. "
    "App có cơ chế fallback để không bị sập."
)

st.sidebar.caption(
    "⚠️ Chỉ dùng để tham khảo, không phải lời khuyên đầu tư."
)


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/140 Safari/537.36"
        ),
        "Accept": "application/json",
    })
    return session


SESSION = get_http_session()


# ============================================================
# BINANCE REQUEST
# ============================================================

def binance_request(path, params=None):
    """
    Thử lần lượt nhiều Binance Futures endpoint.
    Trả về data + endpoint đang hoạt động + lỗi cuối.
    """

    last_error = None

    for base in BINANCE_FUTURES_BASES:

        url = f"{base}{path}"

        try:
            response = SESSION.get(
                url,
                params=params or {},
                timeout=12
            )

            # Không retry các lỗi logic kiểu 400 nếu endpoint trả được
            if response.status_code == 200:
                return response.json(), base, None

            last_error = (
                f"{base} -> HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        except requests.RequestException as exc:
            last_error = f"{base} -> {exc}"

        except Exception as exc:
            last_error = f"{base} -> {exc}"

    return None, None, last_error


# ============================================================
# BINANCE KLINES
# ============================================================

def get_binance_klines(
    symbol="CLUSDT",
    interval="1h",
    limit=200
):
    data, endpoint, error = binance_request(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )

    if data is None:
        return None, None, error

    try:
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]

        df = pd.DataFrame(data, columns=columns)

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote"
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df["open_time"] = pd.to_datetime(
            df["open_time"],
            unit="ms",
            utc=True
        )

        df["close_time"] = pd.to_datetime(
            df["close_time"],
            unit="ms",
            utc=True
        )

        df = df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        if df.empty:
            return None, None, "Binance trả dữ liệu rỗng."

        return df, endpoint, None

    except Exception as exc:
        return None, endpoint, f"Lỗi xử lý nến Binance: {exc}"


# ============================================================
# BINANCE CURRENT PRICE
# ============================================================

def get_binance_price(symbol="CLUSDT"):
    data, endpoint, error = binance_request(
        "/fapi/v1/ticker/price",
        {"symbol": symbol}
    )

    if data is None:
        return None, endpoint, error

    try:
        return float(data["price"]), endpoint, None
    except Exception as exc:
        return None, endpoint, f"Lỗi giá Binance: {exc}"


# ============================================================
# BINANCE FUNDING
# ============================================================

def get_binance_funding(symbol="CLUSDT"):
    data, endpoint, error = binance_request(
        "/fapi/v1/fundingRate",
        {
            "symbol": symbol,
            "limit": 1
        }
    )

    if data is None:
        return None, endpoint, error

    try:
        if not data:
            return None, endpoint, "Không có funding."

        item = data[-1]

        return {
            "rate": float(item["fundingRate"]),
            "time": pd.to_datetime(
                item["fundingTime"],
                unit="ms",
                utc=True
            )
        }, endpoint, None

    except Exception as exc:
        return None, endpoint, f"Lỗi funding: {exc}"


# ============================================================
# BINANCE OPEN INTEREST
# ============================================================

def get_binance_open_interest(symbol="CLUSDT"):
    data, endpoint, error = binance_request(
        "/fapi/v1/openInterest",
        {"symbol": symbol}
    )

    if data is None:
        return None, endpoint, error

    try:
        return float(data["openInterest"]), endpoint, None
    except Exception as exc:
        return None, endpoint, f"Lỗi Open Interest: {exc}"


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        14,
        min_periods=14
    ).mean()

    avg_loss = loss.rolling(
        14,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)

    df["rsi"] = (
        100 -
        (100 / (1 + rs))
    )

    df["rsi"] = df["rsi"].fillna(50)

    # EMA
    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["histogram"] = (
        df["macd"] -
        df["signal"]
    )

    return df


# ============================================================
# YAHOO FALLBACK
# ============================================================

def get_yahoo_fallback(
    interval="1h",
    limit=200
):
    """
    Fallback khi Binance bị HTTP 451 hoặc bị chặn.
    Đây là WTI futures CL=F, KHÔNG phải CLUSDT Binance.
    """

    period_map = {
        "5m": "10d",
        "15m": "60d",
        "30m": "60d",
        "1h": "730d",
        "4h": "730d",
        "1d": "max"
    }

    interval_map = {
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "1h",
        "1d": "1d"
    }

    try:

        raw = yf.download(
            "CL=F",
            period=period_map.get(
                interval,
                "730d"
            ),
            interval=interval_map.get(
                interval,
                "1h"
            ),
            auto_adjust=False,
            progress=False
        )

        if raw is None or raw.empty:
            return None, "Yahoo Finance không có dữ liệu."

        # MultiIndex
        if isinstance(
            raw.columns,
            pd.MultiIndex
        ):
            raw.columns = (
                raw.columns
                .get_level_values(0)
            )

        raw.columns = [
            str(c).lower()
            for c in raw.columns
        ]

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        if not all(
            col in raw.columns
            for col in required
        ):
            return None, "Yahoo thiếu OHLCV."

        df = raw[required].copy()

        for col in required:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna()

        if interval == "4h":
            # resample 1h -> 4h
            df = df.resample("4h").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()

        df = df.tail(limit)

        if df.empty:
            return None, "Yahoo không đủ dữ liệu."

        df.index = pd.to_datetime(
            df.index,
            utc=True
        )

        df = df.reset_index(
            names="open_time"
        )

        return df, None

    except Exception as exc:
        return None, f"Yahoo fallback lỗi: {exc}"


# ============================================================
# NEWS
# ============================================================

def get_global_news(max_results=12):

    queries = [
        "WTI crude oil",
        "OPEC oil",
        "oil inventory",
        "oil geopolitics",
        "WTI oil forecast"
    ]

    articles = []

    try:

        with DDGS() as ddgs:

            for query in queries:

                try:
                    results = ddgs.news(
                        query=query,
                        max_results=4
                    )

                    for item in results:

                        title = item.get("title")

                        if title:
                            articles.append(
                                title.strip()
                            )

                except Exception:
                    continue

        articles = list(
            dict.fromkeys(articles)
        )

        if not articles:
            return [
                "Không lấy được tin tức hiện tại."
            ]

        return articles[:max_results]

    except Exception as exc:
        return [
            f"Không thể lấy tin tức: {exc}"
        ]


# ============================================================
# TECHNICAL SUMMARY
# ============================================================

def get_technical_summary(df):

    last = df.iloc[-1]

    price = float(last["close"])
    rsi = float(last["rsi"])
    macd = float(last["macd"])
    signal = float(last["signal"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])

    score = 0

    reasons = []

    if rsi >= 70:
        score -= 2
        reasons.append(
            "RSI trên 70: thị trường đang quá mua."
        )

    elif rsi <= 30:
        score += 2
        reasons.append(
            "RSI dưới 30: thị trường đang quá bán."
        )

    elif rsi >= 50:
        score += 1
        reasons.append(
            "RSI trên 50: động lượng nghiêng tăng."
        )

    else:
        score -= 1
        reasons.append(
            "RSI dưới 50: động lượng nghiêng giảm."
        )

    if macd > signal:
        score += 2
        reasons.append(
            "MACD trên Signal: động lượng tăng."
        )
    else:
        score -= 2
        reasons.append(
            "MACD dưới Signal: động lượng giảm."
        )

    if price > ema20:
        score += 1
        reasons.append(
            "Giá nằm trên EMA20."
        )
    else:
        score -= 1
        reasons.append(
            "Giá nằm dưới EMA20."
        )

    if price > ema50:
        score += 1
        reasons.append(
            "Giá nằm trên EMA50."
        )
    else:
        score -= 1
        reasons.append(
            "Giá nằm dưới EMA50."
        )

    if score >= 4:
        direction = "TĂNG MẠNH"
    elif score >= 2:
        direction = "TĂNG"
    elif score <= -4:
        direction = "GIẢM MẠNH"
    elif score <= -2:
        direction = "GIẢM"
    else:
        direction = "ĐI NGANG"

    return {
        "price": price,
        "rsi": rsi,
        "macd": macd,
        "signal": signal,
        "ema20": ema20,
        "ema50": ema50,
        "score": score,
        "direction": direction,
        "reasons": reasons
    }


# ============================================================
# GEMINI
# ============================================================

def ai_analyze(
    api_key,
    news,
    technical,
    funding_rate,
    open_interest,
    source_name
):

    try:

        client = genai.Client(
            api_key=api_key
        )

        news_text = "\n".join(
            f"- {x}"
            for x in news
        )

        if funding_rate is None:
            funding_rate_text = "Không có dữ liệu"
        else:
            funding_rate_text = (
                f"{funding_rate * 100:.4f}%"
            )

        if open_interest is None:
            oi_text = "Không có dữ liệu"
        else:
            oi_text = f"{open_interest:,.2f}"

        prompt = f"""
Bạn là chuyên gia phân tích WTI Crude Oil.

Nguồn dữ liệu:
{source_name}

Lưu ý:
Nếu nguồn là fallback Yahoo Finance thì KHÔNG được gọi đây
là dữ liệu CLUSDT Binance. Hãy phân biệt rõ WTI futures và
Binance CLUSDT.

================================
DỮ LIỆU
================================

Giá:
${technical['price']:.2f}

RSI:
{technical['rsi']:.2f}

MACD:
{technical['macd']:.4f}

Signal:
{technical['signal']:.4f}

EMA20:
{technical['ema20']:.2f}

EMA50:
{technical['ema50']:.2f}

Điểm kỹ thuật:
{technical['score']}

Xu hướng kỹ thuật:
{technical['direction']}

Funding Rate:
{funding_rate_text}

Open Interest:
{oi_text}

================================
TIN TỨC
================================

{news_text}

================================
YÊU CẦU
================================

Phân tích bằng tiếng Việt.

Trả lời:

🎯 TÍN HIỆU
LONG / SHORT / WAIT

📈 XU HƯỚNG
Tăng / Giảm / Đi ngang

📊 KỸ THUẬT
RSI + MACD + EMA + giá.

💰 FUTURES SENTIMENT
Funding + Open Interest nếu có dữ liệu.

📰 VĨ MÔ
OPEC + cung cầu + tồn kho + địa chính trị.

💡 KỊCH BẢN
Điều kiện LONG, SHORT và WAIT.

⚠️ RỦI RO
Các yếu tố có thể làm nhận định sai.

Không được khẳng định chắc chắn giá sẽ tăng hoặc giảm.
Đây chỉ là phân tích tham khảo.
"""

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "thinking_level": "low"
            }
        )

        if hasattr(
            interaction,
            "output_text"
        ) and interaction.output_text:

            return interaction.output_text

        return (
            "Gemini không trả về nội dung."
        )

    except Exception as exc:

        return (
            "❌ GEMINI ERROR\n\n"
            f"{exc}"
        )


# ============================================================
# CANDLESTICK CHART
# ============================================================

def build_candlestick_chart(df, symbol_label):

    plot_df = df.tail(150).copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28]
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=plot_df["open_time"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Nến"
        ),
        row=1,
        col=1
    )

    # EMA20
    fig.add_trace(
        go.Scatter(
            x=plot_df["open_time"],
            y=plot_df["ema20"],
            mode="lines",
            name="EMA20"
        ),
        row=1,
        col=1
    )

    # EMA50
    fig.add_trace(
        go.Scatter(
            x=plot_df["open_time"],
            y=plot_df["ema50"],
            mode="lines",
            name="EMA50"
        ),
        row=1,
        col=1
    )

    # Volume
    fig.add_trace(
        go.Bar(
            x=plot_df["open_time"],
            y=plot_df["volume"],
            name="Volume"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        title=f"{symbol_label} — Biểu đồ nến",
        height=650,
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        )
    )

    fig.update_yaxes(
        title_text="Giá",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Volume",
        row=2,
        col=1
    )

    return fig


# ============================================================
# MAIN
# ============================================================

if st.button(
    "🚀 PHÂN TÍCH DẦU",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # API KEY
    # ========================================================

    if not api_key.strip():

        st.error(
            "⚠️ Bạn chưa nhập Gemini API Key."
        )

        st.stop()

    # ========================================================
    # STEP 1: BINANCE
    # ========================================================

    with st.spinner(
        "1/5: Đang kết nối Binance Futures..."
    ):

        binance_df, active_endpoint, binance_error = (
            get_binance_klines(
                BINANCE_SYMBOL,
                interval,
                limit
            )
        )

    source_name = ""
    funding_rate = None
    open_interest = None

    # ========================================================
    # BINANCE SUCCESS
    # ========================================================

    if binance_df is not None:

        source_name = (
            f"Binance Futures {BINANCE_SYMBOL}"
            f" — {active_endpoint}"
        )

        df = binance_df

        # Funding
        funding_data, _, _ = get_binance_funding(
            BINANCE_SYMBOL
        )

        if funding_data:
            funding_rate = funding_data["rate"]

        # OI
        open_interest, _, _ = get_binance_open_interest(
            BINANCE_SYMBOL
        )

        st.success(
            f"✅ Đang dùng dữ liệu Binance Futures "
            f"từ {active_endpoint}"
        )

    # ========================================================
    # BINANCE FAILED → FALLBACK
    # ========================================================

    else:

        st.warning(
            "⚠️ Binance Futures không truy cập được từ "
            "máy chủ Streamlit Cloud."
        )

        with st.expander(
            "Xem lỗi Binance"
        ):
            st.code(
                str(binance_error)
            )

        with st.spinner(
            "Đang chuyển sang dữ liệu WTI dự phòng..."
        ):

            fallback_df, fallback_error = (
                get_yahoo_fallback(
                    interval,
                    limit
                )
            )

        if fallback_df is None:

            st.error(
                "❌ Binance và nguồn dự phòng đều lỗi.\n\n"
                f"Yahoo: {fallback_error}\n\n"
                "Bạn có thể cần đổi nơi deploy app "
                "sang server có quyền truy cập Binance."
            )

            st.stop()

        df = fallback_df

        source_name = (
            "Yahoo Finance WTI Futures (CL=F) — FALLBACK"
        )

        st.warning(
            "🟡 FALLBACK: Đang dùng WTI Futures từ Yahoo Finance. "
            "Đây KHÔNG phải dữ liệu CLUSDT Binance."
        )

    # ========================================================
    # INDICATORS
    # ========================================================

    with st.spinner(
        "2/5: Đang tính RSI, MACD và EMA..."
    ):

        df = add_indicators(
            df
        )

        technical = get_technical_summary(
            df
        )

    # ========================================================
    # NEWS
    # ========================================================

    with st.spinner(
        "3/5: Đang thu thập tin tức dầu mỏ..."
    ):

        news = get_global_news()

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.subheader(
        "📊 Tổng quan"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💵 Giá",
        f"${technical['price']:.2f}"
    )

    c2.metric(
        "RSI",
        f"{technical['rsi']:.2f}"
    )

    c3.metric(
        "MACD",
        (
            "🟢 Tăng"
            if technical["macd"] >
            technical["signal"]
            else "🔴 Giảm"
        )
    )

    c4.metric(
        "Xu hướng kỹ thuật",
        technical["direction"]
    )

    # ========================================================
    # FUTURES
    # ========================================================

    if active_endpoint is not None:

        st.subheader(
            "📡 Binance Futures"
        )

        f1, f2, f3 = st.columns(3)

        if funding_rate is not None:

            f1.metric(
                "Funding Rate",
                f"{funding_rate * 100:.4f}%"
            )

        else:

            f1.metric(
                "Funding Rate",
                "N/A"
            )

        if open_interest is not None:

            f2.metric(
                "Open Interest",
                f"{open_interest:,.2f}"
            )

        else:

            f2.metric(
                "Open Interest",
                "N/A"
            )

        f3.metric(
            "Nguồn",
            "CLUSDT"
        )

    # ========================================================
    # CANDLESTICK
    # ========================================================

    st.subheader(
        "🕯️ Biểu đồ nến"
    )

    fig = build_candlestick_chart(
        df,
        (
            "CLUSDT"
            if active_endpoint
            else "WTI CL=F (Fallback)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True
        }
    )

    # ========================================================
    # RSI
    # ========================================================

    st.subheader(
        "📈 RSI"
    )

    rsi_fig = go.Figure()

    rsi_fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["rsi"],
            mode="lines",
            name="RSI"
        )
    )

    rsi_fig.add_hline(
        y=70,
        line_dash="dash"
    )

    rsi_fig.add_hline(
        y=30,
        line_dash="dash"
    )

    rsi_fig.update_layout(
        height=300,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        )
    )

    st.plotly_chart(
        rsi_fig,
        use_container_width=True,
        config={"displaylogo": False}
    )

    # ========================================================
    # MACD
    # ========================================================

    st.subheader(
        "📉 MACD"
    )

    macd_fig = go.Figure()

    macd_fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["macd"],
            mode="lines",
            name="MACD"
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["signal"],
            mode="lines",
            name="Signal"
        )
    )

    macd_fig.update_layout(
        height=300,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        )
    )

    st.plotly_chart(
        macd_fig,
        use_container_width=True,
        config={"displaylogo": False}
    )

    # ========================================================
    # TECHNICAL REASONS
    # ========================================================

    st.subheader(
        "🧠 Tín hiệu kỹ thuật"
    )

    for reason in technical["reasons"]:

        st.write(
            "• " + reason
        )

    # ========================================================
    # AI
    # ========================================================

    with st.spinner(
        "4/5: Gemini đang phân tích toàn bộ dữ liệu..."
    ):

        analysis = ai_analyze(
            api_key=api_key.strip(),
            news=news,
            technical=technical,
            funding_rate=funding_rate,
            open_interest=open_interest,
            source_name=source_name
        )

    st.subheader(
        "🤖 Báo cáo AI"
    )

    st.markdown(
        analysis
    )

    # ========================================================
    # NEWS
    # ========================================================

    st.subheader(
        "📰 Tin tức đầu vào"
    )

    with st.expander(
        "Xem các tin tức đã thu thập"
    ):

        for i, item in enumerate(
            news,
            start=1
        ):

            st.write(
                f"**{i}.** {item}"
            )

    # ========================================================
    # STATUS
    # ========================================================

    st.markdown("---")

    st.caption(
        f"Nguồn dữ liệu: {source_name}"
    )

    st.caption(
        "⚠️ Hệ thống chỉ cung cấp phân tích tham khảo, "
        "không phải lời khuyên đầu tư."
    )
