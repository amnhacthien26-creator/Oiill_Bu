import io
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ddgs import DDGS
from google import genai


# ============================================================
# CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="AI Phân tích Dầu CLUSDT",
    page_icon="🛢️",
    layout="wide",
)

SYMBOL = "CLUSDT"

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_PRO_BASE = "https://pro-api.coingecko.com/api/v3"

BINANCE_DATA_BASE = "https://data.binance.vision"

st.title("🛢️ AI Phân tích Giá Dầu CLUSDT")
st.markdown(
    """
    Phân tích WTI trên Binance Futures bằng:
    **CoinGecko → giá/Funding/Open Interest/Volume**
    + **Binance Public Data → nến lịch sử**
    + **RSI / MACD / EMA**
    + **Tin tức + Gemini AI**
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Cấu hình")

gemini_api_key = st.sidebar.text_input(
    "🔑 Gemini API Key",
    type="password",
    placeholder="Dán Gemini API Key",
)

coingecko_api_key = st.sidebar.text_input(
    "🦎 CoinGecko API Key",
    type="password",
    placeholder="Có thể để trống nếu endpoint public hoạt động",
)

st.sidebar.markdown(
    "[👉 CoinGecko API](https://www.coingecko.com/api)"
)

st.sidebar.markdown(
    "[👉 Gemini API](https://aistudio.google.com/apikey)"
)

st.sidebar.markdown("---")

interval = st.sidebar.selectbox(
    "⏱️ Khung thời gian",
    ["5m", "15m", "30m", "1h", "4h", "1d"],
    index=3,
)

limit = st.sidebar.selectbox(
    "📊 Số nến",
    [100, 200, 300, 500],
    index=1,
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Symbol: **CLUSDT**
    
    Binance Futures:
    **WTI Crude Oil**
    
    CoinGecko:
    dữ liệu Futures trung gian.
    """
)

st.sidebar.caption(
    "⚠️ Chỉ dùng để tham khảo, không phải lời khuyên đầu tư."
)


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_session():
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/140 Safari/537.36"
            ),
            "Accept": "application/json",
        }
    )

    return session


SESSION = get_session()


# ============================================================
# COINGECKO REQUEST
# ============================================================

def coingecko_get(
    path,
    params=None,
):
    """
    Ưu tiên Pro API nếu có API key.
    Nếu không có, thử public API.
    """

    headers = {}

    if coingecko_api_key.strip():
        base_url = COINGECKO_PRO_BASE
        headers["x-cg-pro-api-key"] = (
            coingecko_api_key.strip()
        )
    else:
        base_url = COINGECKO_BASE

    try:

        response = SESSION.get(
            base_url + path,
            params=params or {},
            headers=headers,
            timeout=20,
        )

        if response.status_code == 200:
            return response.json(), None

        return None, (
            f"CoinGecko HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    except Exception as exc:

        return None, (
            f"Lỗi CoinGecko: {exc}"
        )


# ============================================================
# TÌM CLUSDT TRÊN BINANCE FUTURES COINGECKO
# ============================================================

def get_clusdt_from_coingecko():
    """
    Lấy dữ liệu derivatives của CLUSDT.

    CoinGecko trả:
    - price
    - index
    - basis
    - funding_rate
    - open_interest
    - volume_24h
    - last_traded_at
    """

    # Cách 1: derivatives/exchanges/binance_futures
    data, error = coingecko_get(
        "/derivatives/exchanges/binance_futures",
        {
            "include_tickers": "all",
        },
    )

    if data is not None:

        tickers = data.get(
            "tickers",
            [],
        )

        for ticker in tickers:

            symbol = str(
                ticker.get("symbol", "")
            ).upper()

            if symbol == SYMBOL:

                return {
                    "symbol": symbol,
                    "market": "Binance (Futures)",
                    "price": safe_number(
                        ticker.get("last")
                    ),
                    "index": safe_number(
                        ticker.get("index")
                    ),
                    "basis": safe_number(
                        ticker.get(
                            "index_basis_percentage"
                        )
                    ),
                    "funding_rate": safe_number(
                        ticker.get("funding_rate")
                    ),
                    "open_interest": safe_number(
                        ticker.get(
                            "open_interest_usd"
                        )
                    ),
                    "volume_24h": safe_number(
                        ticker.get(
                            "h24_volume"
                        )
                    ),
                    "last_traded": ticker.get(
                        "last_traded"
                    ),
                    "trade_url": ticker.get(
                        "trade_url"
                    ),
                }, None

    # Cách 2: /derivatives
    data2, error2 = coingecko_get(
        "/derivatives",
    )

    if data2 is not None:

        for ticker in data2:

            if (
                str(
                    ticker.get("market", "")
                ).lower()
                .startswith("binance")
            ):

                symbol = str(
                    ticker.get("symbol", "")
                ).upper()

                if symbol == SYMBOL:

                    return {
                        "symbol": symbol,
                        "market": ticker.get(
                            "market",
                            "Binance (Futures)",
                        ),
                        "price": safe_number(
                            ticker.get("price")
                        ),
                        "index": safe_number(
                            ticker.get("index")
                        ),
                        "basis": safe_number(
                            ticker.get("basis")
                        ),
                        "funding_rate": safe_number(
                            ticker.get(
                                "funding_rate"
                            )
                        ),
                        "open_interest": safe_number(
                            ticker.get(
                                "open_interest"
                            )
                        ),
                        "volume_24h": safe_number(
                            ticker.get(
                                "volume_24h"
                            )
                        ),
                        "last_traded": ticker.get(
                            "last_traded_at"
                        ),
                        "trade_url": None,
                    }, None

    return None, (
        error
        or error2
        or "Không tìm thấy CLUSDT trên CoinGecko."
    )


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_number(value):
    try:

        if value is None:
            return None

        if isinstance(
            value,
            str,
        ):
            value = value.replace(
                ",",
                "",
            )

        number = float(value)

        if pd.isna(number):
            return None

        return number

    except Exception:
        return None


# ============================================================
# BINANCE PUBLIC DATA ARCHIVE
# ============================================================

def get_archive_url(
    symbol,
    interval_name,
    date_obj,
):
    date_text = (
        date_obj.strftime(
            "%Y-%m-%d"
        )
    )

    return (
        f"{BINANCE_DATA_BASE}/data/futures/um/daily/"
        f"klines/{symbol}/{interval_name}/"
        f"{symbol}-{interval_name}-{date_text}.zip"
    )


# ============================================================
# TẢI 1 NGÀY NẾN
# ============================================================

def download_archive_day(
    symbol,
    interval_name,
    date_obj,
):

    url = get_archive_url(
        symbol,
        interval_name,
        date_obj,
    )

    try:

        response = SESSION.get(
            url,
            timeout=20,
        )

        if response.status_code != 200:
            return None, (
                f"HTTP {response.status_code}"
            )

        if not response.content:
            return None, "File rỗng."

        with zipfile.ZipFile(
            io.BytesIO(
                response.content
            )
        ) as archive:

            csv_files = [
                x
                for x in archive.namelist()
                if x.endswith(".csv")
            ]

            if not csv_files:
                return None, (
                    "ZIP không chứa CSV."
                )

            with archive.open(
                csv_files[0]
            ) as csv_file:

                data = pd.read_csv(
                    csv_file,
                    header=None,
                )

        if data.empty:
            return None, "CSV rỗng."

        # Binance Kline CSV
        data = data.iloc[:, :12]

        data.columns = [
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
            "ignore",
        ]

        # Nếu archive có header
        first_cell = str(
            data.iloc[0, 0]
        ).lower()

        if "open_time" in first_cell:

            data = data.iloc[1:].copy()

        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]

        for col in numeric_cols:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

        data["open_time"] = pd.to_datetime(
            data["open_time"],
            unit="ms",
            utc=True,
            errors="coerce",
        )

        data = data.dropna(
            subset=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        return data, None

    except Exception as exc:

        return None, str(exc)


# ============================================================
# LẤY LỊCH SỬ NẾN
# ============================================================

def get_binance_candles(
    interval_name,
    limit_count,
):
    """
    Không dùng fapi.binance.com.

    Lấy từ Binance Public Data Archive.
    """

    source_interval = interval_name

    # 4h: lấy 1h rồi resample
    if interval_name == "4h":
        source_interval = "1h"

    bars_per_day = {
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "1d": 1,
    }

    estimated_days = int(
        limit_count /
        bars_per_day[source_interval]
    ) + 2

    estimated_days = max(
        2,
        estimated_days,
    )

    estimated_days = min(
        estimated_days,
        14,
    )

    today = datetime.now(
        timezone.utc
    ).date()

    frames = []

    for offset in range(
        estimated_days + 3
    ):

        date_obj = (
            today -
            timedelta(
                days=offset
            )
        )

        frame, error = (
            download_archive_day(
                SYMBOL,
                source_interval,
                date_obj,
            )
        )

        if frame is not None:
            frames.append(frame)

        total_rows = sum(
            len(x)
            for x in frames
        )

        if total_rows >= (
            limit_count + 50
        ):
            break

    if not frames:

        return None, (
            "Không thể lấy nến CLUSDT "
            "từ Binance Public Data Archive."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = (
        df.sort_values(
            "open_time"
        )
        .drop_duplicates(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )

    # 4h
    if interval_name == "4h":

        df = (
            df.set_index(
                "open_time"
            )
            .resample("4h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )

    df = df.tail(
        limit_count
    )

    if len(df) < 30:

        return None, (
            f"Chỉ lấy được {len(df)} nến."
        )

    return df, None


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(
    df,
):

    df = df.copy()

    # RSI
    delta = (
        df["close"]
        .diff()
    )

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14,
        min_periods=14,
    ).mean()

    avg_loss = loss.rolling(
        14,
        min_periods=14,
    ).mean()

    avg_loss = avg_loss.replace(
        0,
        pd.NA,
    )

    rs = (
        avg_gain /
        avg_loss
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (
                1 + rs
            )
        )
    )

    df["rsi"] = (
        df["rsi"]
        .fillna(50)
    )

    # EMA
    df["ema20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    # MACD
    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    df["macd"] = (
        ema12 -
        ema26
    )

    df["signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["histogram"] = (
        df["macd"] -
        df["signal"]
    )

    return df


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_analysis(
    df,
):

    last = df.iloc[-1]

    price = float(
        last["close"]
    )

    rsi = float(
        last["rsi"]
    )

    macd = float(
        last["macd"]
    )

    signal = float(
        last["signal"]
    )

    ema20 = float(
        last["ema20"]
    )

    ema50 = float(
        last["ema50"]
    )

    score = 0

    reasons = []

    # RSI
    if rsi <= 30:

        score += 2

        reasons.append(
            "RSI dưới 30: thị trường đang quá bán."
        )

    elif rsi >= 70:

        score -= 2

        reasons.append(
            "RSI trên 70: thị trường đang quá mua."
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

    # MACD
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

    # EMA20
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

    # EMA50
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
        "reasons": reasons,
    }


# ============================================================
# NEWS
# ============================================================

def get_news(
    max_results=15,
):

    queries = [
        "WTI crude oil",
        "OPEC oil",
        "oil inventory",
        "oil geopolitics",
        "WTI oil price",
    ]

    all_news = []

    try:

        with DDGS() as ddgs:

            for query in queries:

                try:

                    results = ddgs.news(
                        query=query,
                        max_results=4,
                    )

                    for item in results:

                        title = item.get(
                            "title"
                        )

                        if title:

                            all_news.append(
                                title.strip()
                            )

                except Exception:
                    continue

        unique_news = list(
            dict.fromkeys(
                all_news
            )
        )

        if not unique_news:

            return [
                "Không lấy được tin tức mới."
            ]

        return unique_news[
            :max_results
        ]

    except Exception as exc:

        return [
            f"Không thể lấy tin tức: {exc}"
        ]


# ============================================================
# GEMINI
# ============================================================

def analyze_with_gemini(
    api_key,
    ticker,
    technical,
    futures,
    news,
):

    try:

        client = genai.Client(
            api_key=api_key
        )

        funding = futures.get(
            "funding_rate"
        )

        oi = futures.get(
            "open_interest"
        )

        volume = futures.get(
            "volume_24h"
        )

        index_price = futures.get(
            "index"
        )

        basis = futures.get(
            "basis"
        )

        if funding is None:
            funding_text = "Không có dữ liệu"
        else:
            funding_text = (
                f"{funding * 100:.4f}%"
            )

        if oi is None:
            oi_text = "Không có dữ liệu"
        else:
            oi_text = (
                f"${oi:,.0f}"
            )

        if volume is None:
            volume_text = "Không có dữ liệu"
        else:
            volume_text = (
                f"${volume:,.0f}"
            )

        if index_price is None:
            index_text = "Không có dữ liệu"
        else:
            index_text = (
                f"${index_price:.2f}"
            )

        if basis is None:
            basis_text = "Không có dữ liệu"
        else:
            basis_text = (
                f"{basis:.4f}%"
            )

        news_text = "\n".join(
            f"- {x}"
            for x in news
        )

        prompt = f"""
Bạn là chuyên gia phân tích dầu WTI
trên Binance Futures.

Dữ liệu Futures hiện tại lấy qua CoinGecko.

SYMBOL:
{ticker}

GIÁ:
${technical['price']:.2f}

INDEX PRICE:
{index_text}

BASIS:
{basis_text}

RSI:
{technical['rsi']:.2f}

MACD:
{technical['macd']:.4f}

SIGNAL:
{technical['signal']:.4f}

EMA20:
${technical['ema20']:.2f}

EMA50:
${technical['ema50']:.2f}

ĐIỂM KỸ THUẬT:
{technical['score']}

XU HƯỚNG:
{technical['direction']}

FUNDING RATE:
{funding_text}

OPEN INTEREST:
{oi_text}

VOLUME 24H:
{volume_text}

TIN TỨC:
{news_text}

Hãy trả lời bằng tiếng Việt.

Cấu trúc:

🎯 TÍN HIỆU
LONG / SHORT / WAIT

📈 XU HƯỚNG
Tăng / Giảm / Đi ngang

📊 KỸ THUẬT
Phân tích RSI, MACD, EMA20 và EMA50.

💰 FUTURES SENTIMENT
Phân tích Funding Rate,
Open Interest và Volume.

📰 VĨ MÔ
Phân tích OPEC, cung cầu,
tồn kho và địa chính trị.

💡 KỊCH BẢN
LONG:
Điều kiện.

SHORT:
Điều kiện.

WAIT:
Điều kiện.

⚠️ RỦI RO
Những yếu tố có thể làm tín hiệu sai.

QUY TẮC:

- Không khẳng định chắc chắn giá.
- Không tự tạo dữ liệu.
- Không dùng HTML.
- Không dùng ký tự lỗi mã hóa.
- Markdown đơn giản.
- Không gọi CL=F là CLUSDT.
- Đây chỉ là phân tích tham khảo.
"""

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "thinking_level": "low",
            },
        )

        if (
            hasattr(
                interaction,
                "output_text",
            )
            and interaction.output_text
        ):

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
# CANDLE CHART
# ============================================================

def candle_chart(
    df,
):

    chart_df = df.tail(
        200
    ).copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[
            0.75,
            0.25,
        ],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df[
                "open_time"
            ],
            open=chart_df[
                "open"
            ],
            high=chart_df[
                "high"
            ],
            low=chart_df[
                "low"
            ],
            close=chart_df[
                "close"
            ],
            name=SYMBOL,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df[
                "open_time"
            ],
            y=chart_df[
                "ema20"
            ],
            name="EMA20",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df[
                "open_time"
            ],
            y=chart_df[
                "ema50"
            ],
            name="EMA50",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=chart_df[
                "open_time"
            ],
            y=chart_df[
                "volume"
            ],
            name="Volume",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=650,
        title=(
            f"{SYMBOL} — Biểu đồ nến "
            f"({interval})"
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10,
        ),
    )

    return fig


# ============================================================
# RSI CHART
# ============================================================

def rsi_chart(
    df,
):

    chart_df = df.tail(
        200
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df[
                "open_time"
            ],
            y=chart_df[
                "rsi"
            ],
            name="RSI",
            mode="lines",
        )
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
    )

    fig.add_hline(
        y=50,
        line_dash="dot",
    )

    fig.update_layout(
        height=300,
        title="RSI(14)",
        yaxis=dict(
            range=[0, 100]
        ),
    )

    return fig


# ============================================================
# MACD CHART
# ============================================================

def macd_chart(
    df,
):

    chart_df = df.tail(
        200
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df[
                "open_time"
            ],
            y=chart_df[
                "macd"
            ],
            name="MACD",
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df[
                "open_time"
            ],
            y=chart_df[
                "signal"
            ],
            name="Signal",
            mode="lines",
        )
    )

    fig.add_bar(
        x=chart_df[
            "open_time"
        ],
        y=chart_df[
            "histogram"
        ],
        name="Histogram",
    )

    fig.update_layout(
        height=320,
        title="MACD(12,26,9)",
    )

    return fig


# ============================================================
# MAIN
# ============================================================

if st.button(
    "🚀 PHÂN TÍCH CLUSDT",
    type="primary",
    use_container_width=True,
):

    # ========================================================
    # KIỂM TRA GEMINI
    # ========================================================

    if not gemini_api_key.strip():

        st.error(
            "⚠️ Vui lòng nhập Gemini API Key."
        )

        st.stop()

    # ========================================================
    # COINGECKO
    # ========================================================

    with st.spinner(
        "1/5: Đang lấy dữ liệu CLUSDT từ CoinGecko..."
    ):

        futures, cg_error = (
            get_clusdt_from_coingecko()
        )

    if futures is None:

        st.error(
            "❌ Không lấy được CLUSDT từ CoinGecko."
        )

        st.code(
            str(cg_error)
        )

        st.stop()

    # ========================================================
    # NẾN
    # ========================================================

    with st.spinner(
        "2/5: Đang lấy dữ liệu nến CLUSDT..."
    ):

        candles, candle_error = (
            get_binance_candles(
                interval,
                limit,
            )
        )

    if candles is None:

        st.error(
            "❌ Không lấy được dữ liệu nến CLUSDT."
        )

        st.code(
            str(candle_error)
        )

        st.info(
            "CoinGecko cung cấp dữ liệu Futures "
            "hiện tại nhưng không có OHLC lịch sử "
            "riêng cho CLUSDT trong endpoint derivatives. "
            "App không tạo nến giả."
        )

        st.stop()

    # ========================================================
    # INDICATORS
    # ========================================================

    with st.spinner(
        "3/5: Đang tính RSI, MACD và EMA..."
    ):

        candles = add_indicators(
            candles
        )

        if len(candles) < 30:

            st.error(
                "Không đủ nến để tính chỉ báo."
            )

            st.stop()

        technical = (
            technical_analysis(
                candles
            )
        )

    # ========================================================
    # NEWS
    # ========================================================

    with st.spinner(
        "4/5: Đang lấy tin tức dầu mỏ..."
    ):

        news = get_news()

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.subheader(
        "📊 Binance CLUSDT"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    # Ưu tiên giá CoinGecko
    live_price = (
        futures["price"]
        if futures["price"] is not None
        else technical["price"]
    )

    c1.metric(
        "💵 Giá CLUSDT",
        f"${live_price:.2f}",
    )

    c2.metric(
        "RSI",
        f"{technical['rsi']:.2f}",
    )

    c3.metric(
        "MACD",
        (
            "🟢 Tăng"
            if technical["macd"]
            > technical["signal"]
            else "🔴 Giảm"
        ),
    )

    c4.metric(
        "Xu hướng kỹ thuật",
        technical["direction"],
    )

    # ========================================================
    # FUTURES DATA
    # ========================================================

    st.subheader(
        "📡 Futures Sentiment"
    )

    f1, f2, f3, f4 = (
        st.columns(4)
    )

    funding = futures[
        "funding_rate"
    ]

    oi = futures[
        "open_interest"
    ]

    vol24 = futures[
        "volume_24h"
    ]

    index_price = futures[
        "index"
    ]

    if funding is not None:

        f1.metric(
            "Funding",
            f"{funding * 100:.4f}%",
        )

    else:

        f1.metric(
            "Funding",
            "N/A",
        )

    if oi is not None:

        f2.metric(
            "Open Interest",
            f"${oi:,.0f}",
        )

    else:

        f2.metric(
            "Open Interest",
            "N/A",
        )

    if vol24 is not None:

        f3.metric(
            "Volume 24h",
            f"${vol24:,.0f}",
        )

    else:

        f3.metric(
            "Volume 24h",
            "N/A",
        )

    if index_price is not None:

        f4.metric(
            "Index Price",
            f"${index_price:.2f}",
        )

    else:

        f4.metric(
            "Index Price",
            "N/A",
        )

    # ========================================================
    # BASIS
    # ========================================================

    basis = futures.get(
        "basis"
    )

    if basis is not None:

        st.caption(
            f"Basis: {basis:.4f}%"
        )

    # ========================================================
    # DATA SOURCE
    # ========================================================

    st.success(
        "✅ Futures hiện tại: CoinGecko → Binance Futures CLUSDT"
    )

    st.info(
        "🕯️ OHLC/nến lịch sử: Binance Public Data Archive. "
        "Không sử dụng fapi.binance.com."
    )

    # ========================================================
    # CANDLE
    # ========================================================

    st.subheader(
        "🕯️ Biểu đồ nến CLUSDT"
    )

    fig = candle_chart(
        candles
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )

    # ========================================================
    # RSI
    # ========================================================

    st.subheader(
        "📈 RSI"
    )

    st.plotly_chart(
        rsi_chart(
            candles
        ),
        use_container_width=True,
        config={
            "displaylogo": False
        },
    )

    # ========================================================
    # MACD
    # ========================================================

    st.subheader(
        "📉 MACD"
    )

    st.plotly_chart(
        macd_chart(
            candles
        ),
        use_container_width=True,
        config={
            "displaylogo": False
        },
    )

    # ========================================================
    # TECHNICAL
    # ========================================================

    st.subheader(
        "🧠 Tín hiệu kỹ thuật"
    )

    for reason in technical[
        "reasons"
    ]:

        st.write(
            f"• {reason}"
        )

    # ========================================================
    # GEMINI
    # ========================================================

    with st.spinner(
        "5/5: Gemini đang tổng hợp dữ liệu..."
    ):

        analysis = (
            analyze_with_gemini(
                api_key=gemini_api_key.strip(),
                ticker=SYMBOL,
                technical=technical,
                futures=futures,
                news=news,
            )
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
        "Xem tin tức"
    ):

        for i, item in enumerate(
            news,
            start=1,
        ):

            st.write(
                f"**{i}.** {item}"
            )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    current_time = datetime.now(
        timezone.utc
    )

    st.caption(
        f"CLUSDT | {interval} | "
        f"Cập nhật: "
        f"{current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    st.caption(
        "⚠️ Dữ liệu và phân tích AI chỉ mang tính tham khảo."
    )
