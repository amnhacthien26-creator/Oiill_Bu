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
    page_title="AI Phân tích CLUSDT Binance",
    page_icon="🛢️",
    layout="wide",
)

SYMBOL = "CLUSDT"

# Binance Futures REST có thể bị chặn trên Streamlit Cloud.
BINANCE_FAPI = "https://fapi.binance.com"

# Binance public data archive.
BINANCE_DATA_BASE = "https://data.binance.vision"

# Số ngày tối đa app thử tải từ archive.
MAX_ARCHIVE_DAYS = 14


# ============================================================
# GIAO DIỆN
# ============================================================

st.title("🛢️ AI Phân tích Dầu WTI — Binance CLUSDT")

st.markdown(
    """
    Phân tích **CLUSDT (WTI Crude Oil)** bằng dữ liệu Binance,
    biểu đồ nến, RSI, MACD, EMA, Volume, Funding Rate,
    Open Interest, tin tức và Gemini AI.
    """
)

st.sidebar.header("⚙️ Cấu hình")

api_key = st.sidebar.text_input(
    "🔑 Gemini API Key",
    type="password",
    placeholder="Dán Gemini API Key vào đây",
)

st.sidebar.markdown(
    "[👉 Lấy Gemini API Key]"
    "(https://aistudio.google.com/apikey)"
)

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
    **Hợp đồng:** CLUSDT  
    **Tài sản:** WTI Crude Oil  
    **Sàn:** Binance Futures
    """
)

st.sidebar.caption(
    "⚠️ Đây là công cụ phân tích tham khảo, "
    "không phải lời khuyên đầu tư."
)


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def create_session():
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/140.0 Safari/537.36"
            ),
            "Accept": "*/*",
        }
    )

    return session


SESSION = create_session()


# ============================================================
# CHUYỂN INTERVAL -> MILLISECONDS
# ============================================================

def interval_to_ms(interval_name):
    mapping = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    return mapping[interval_name]


# ============================================================
# BINANCE REST
# ============================================================

def get_binance_rest_klines(
    symbol,
    interval_name,
    limit_count,
):
    """
    Thử Binance Futures REST.
    Nếu Cloud bị 451 sẽ trả lỗi để chuyển sang archive.
    """

    try:
        response = SESSION.get(
            f"{BINANCE_FAPI}/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": interval_name,
                "limit": limit_count,
            },
            timeout=15,
        )

        if response.status_code != 200:
            return None, (
                f"HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        data = response.json()

        if not data:
            return None, "Binance REST trả dữ liệu rỗng."

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
            "ignore",
        ]

        df = pd.DataFrame(
            data,
            columns=columns,
        )

        return clean_klines(df), None

    except Exception as exc:
        return None, str(exc)


# ============================================================
# CLEAN KLINES
# ============================================================

def clean_klines(df):
    df = df.copy()

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
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    if "open_time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(
            df["open_time"]
        ):
            df["open_time"] = pd.to_datetime(
                df["open_time"],
                unit="ms",
                utc=True,
                errors="coerce",
            )

    df = df.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    df = (
        df.sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# TẠO URL BINANCE DATA ARCHIVE
# ============================================================

def archive_url(
    symbol,
    interval_name,
    date_obj,
):
    date_text = date_obj.strftime("%Y-%m-%d")

    return (
        f"{BINANCE_DATA_BASE}/data/futures/um/daily/"
        f"klines/{symbol}/{interval_name}/"
        f"{symbol}-{interval_name}-{date_text}.zip"
    )


# ============================================================
# DOWNLOAD 1 NGÀY TỪ BINANCE ARCHIVE
# ============================================================

def download_archive_day(
    symbol,
    interval_name,
    date_obj,
):
    url = archive_url(
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
            io.BytesIO(response.content)
        ) as z:

            csv_files = [
                name
                for name in z.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                return None, "ZIP không có CSV."

            with z.open(csv_files[0]) as csv_file:

                raw = pd.read_csv(
                    csv_file,
                    header=None,
                )

        if raw.empty:
            return None, "CSV rỗng."

        expected_cols = [
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

        # Một số archive có header.
        first_value = str(raw.iloc[0, 0]).lower()

        if first_value in (
            "open_time",
            "opentime",
        ):
            raw = raw.iloc[1:].reset_index(
                drop=True
            )

        # Archive Binance có thể có đúng 12 cột.
        if raw.shape[1] < 12:
            return None, (
                f"CSV có {raw.shape[1]} cột, "
                "không đúng định dạng."
            )

        raw = raw.iloc[:, :12]
        raw.columns = expected_cols

        return clean_klines(raw), None

    except zipfile.BadZipFile:
        return None, "File tải về không phải ZIP hợp lệ."

    except Exception as exc:
        return None, str(exc)


# ============================================================
# BINANCE PUBLIC DATA ARCHIVE
# ============================================================

def get_binance_archive_klines(
    symbol,
    interval_name,
    limit_count,
):
    """
    Lấy dữ liệu Binance từ public data archive.

    Ưu điểm:
    - Không phụ thuộc fapi.binance.com.
    - Hữu ích khi Streamlit Cloud bị HTTP 451.

    Với khung 4h, tải 1h rồi resample.
    """

    source_interval = interval_name

    if interval_name == "4h":
        source_interval = "1h"

    # Ước tính cần bao nhiêu ngày dữ liệu
    bars_per_day = {
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "4h": 6,
        "1d": 1,
    }

    per_day = bars_per_day[source_interval]

    days_needed = max(
        2,
        int(limit_count / per_day) + 2,
    )

    days_needed = min(
        days_needed,
        MAX_ARCHIVE_DAYS,
    )

    today = datetime.now(
        timezone.utc
    ).date()

    frames = []
    attempted = 0
    successes = 0

    # Lùi từ hôm nay về trước.
    for offset in range(days_needed + 3):

        if len(frames) > 0:
            current_bars = sum(
                len(frame)
                for frame in frames
            )

            if current_bars >= limit_count + 50:
                break

        date_obj = today - timedelta(
            days=offset
        )

        attempted += 1

        day_df, error = download_archive_day(
            symbol,
            source_interval,
            date_obj,
        )

        if day_df is not None:

            frames.append(day_df)
            successes += 1

    if not frames:
        return None, (
            "Không thể lấy dữ liệu từ Binance "
            "Public Data Archive."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = (
        df.sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )

    # Resample 1h -> 4h
    if interval_name == "4h":

        df = (
            df.set_index("open_time")
            .resample("4h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "quote_volume": "sum",
                    "trades": "sum",
                    "taker_buy_base": "sum",
                    "taker_buy_quote": "sum",
                }
            )
            .dropna(
                subset=[
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            )
            .reset_index()
        )

    df = df.tail(
        limit_count
    ).reset_index(
        drop=True
    )

    if len(df) < 20:
        return None, (
            "Binance Archive trả quá ít nến "
            f"({len(df)} nến)."
        )

    return df, (
        f"Binance Public Data Archive "
        f"({successes}/{attempted} ngày)"
    )


# ============================================================
# TẠO DỮ LIỆU TECHNICAL
# ============================================================

def add_indicators(df):

    df = df.copy()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        window=14,
        min_periods=14,
    ).mean()

    avg_loss = loss.rolling(
        window=14,
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
            (1 + rs)
        )
    )

    df["rsi"] = df["rsi"].fillna(
        50
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
# BINANCE FUNDING
# ============================================================

def get_funding_rate(
    symbol
):
    try:

        response = SESSION.get(
            f"{BINANCE_FAPI}/fapi/v1/fundingRate",
            params={
                "symbol": symbol,
                "limit": 1,
            },
            timeout=10,
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if not data:
            return None

        return float(
            data[-1]["fundingRate"]
        )

    except Exception:
        return None


# ============================================================
# BINANCE OPEN INTEREST
# ============================================================

def get_open_interest(
    symbol
):
    try:

        response = SESSION.get(
            f"{BINANCE_FAPI}/fapi/v1/openInterest",
            params={
                "symbol": symbol,
            },
            timeout=10,
        )

        if response.status_code != 200:
            return None

        data = response.json()

        return float(
            data["openInterest"]
        )

    except Exception:
        return None


# ============================================================
# BINANCE PRICE INDEX / MARK PRICE
# ============================================================

def get_binance_mark_price(
    symbol
):
    try:

        response = SESSION.get(
            f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
            params={
                "symbol": symbol,
            },
            timeout=10,
        )

        if response.status_code != 200:
            return None

        data = response.json()

        return {
            "mark_price": float(
                data["markPrice"]
            ),
            "index_price": float(
                data["indexPrice"]
            ),
            "funding_rate": float(
                data["lastFundingRate"]
            ),
        }

    except Exception:
        return None


# ============================================================
# TIN TỨC
# ============================================================

def get_news(
    max_results=15
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

        all_news = list(
            dict.fromkeys(
                all_news
            )
        )

        if not all_news:
            return [
                "Không tìm thấy tin tức mới."
            ]

        return all_news[
            :max_results
        ]

    except Exception as exc:

        return [
            f"Không thể lấy tin tức: {exc}"
        ]


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_analysis(
    df
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
# GEMINI
# ============================================================

def analyze_with_gemini(
    api_key,
    technical,
    funding_rate,
    open_interest,
    news,
    data_source,
):

    try:

        client = genai.Client(
            api_key=api_key
        )

        if funding_rate is None:

            funding_text = "Không có dữ liệu"

        else:

            funding_text = (
                f"{funding_rate * 100:.4f}%"
            )

        if open_interest is None:

            oi_text = "Không có dữ liệu"

        else:

            oi_text = (
                f"{open_interest:,.2f}"
            )

        news_text = "\n".join(
            f"- {item}"
            for item in news
        )

        prompt = f"""
Bạn là chuyên gia phân tích WTI Crude Oil
trên Binance Futures.

NGUỒN DỮ LIỆU:
{data_source}

SYMBOL:
CLUSDT

GIÁ:
${technical['price']:.2f}

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

XU HƯỚNG KỸ THUẬT:
{technical['direction']}

FUNDING RATE:
{funding_text}

OPEN INTEREST:
{oi_text}

TIN TỨC:
{news_text}

Hãy trả lời bằng tiếng Việt.

Bắt buộc theo cấu trúc:

🎯 TÍN HIỆU
LONG / SHORT / WAIT

📈 XU HƯỚNG
Tăng / Giảm / Đi ngang

📊 KỸ THUẬT
Phân tích RSI, MACD, EMA20, EMA50.

💰 FUTURES SENTIMENT
Phân tích Funding Rate và Open Interest
nếu có dữ liệu.

📰 VĨ MÔ
Phân tích OPEC, cung cầu, tồn kho,
địa chính trị và nhu cầu dầu.

💡 KỊCH BẢN
LONG:
Điều kiện để cân nhắc LONG.

SHORT:
Điều kiện để cân nhắc SHORT.

WAIT:
Khi nào nên đứng ngoài.

⚠️ RỦI RO
Các yếu tố có thể làm nhận định sai.

QUY TẮC:

- Không khẳng định chắc chắn giá tăng hoặc giảm.
- Không tự bịa dữ liệu chưa được cung cấp.
- Nếu dữ liệu Futures bị thiếu, hãy nói rõ.
- Không viết HTML.
- Không sử dụng ký hiệu lạ làm hỏng Markdown.
- Chỉ dùng Markdown đơn giản.
- Đây là phân tích tham khảo, không phải lời khuyên đầu tư.
"""

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "thinking_level": "low"
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
            f"{str(exc)}"
        )


# ============================================================
# BIỂU ĐỒ NẾN
# ============================================================

def create_candle_chart(
    df
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
            name="CLUSDT",
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
            mode="lines",
            name="EMA20",
            line=dict(
                width=1.5
            ),
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
            mode="lines",
            name="EMA50",
            line=dict(
                width=1.5
            ),
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
        title=f"CLUSDT — {interval}",
        height=650,
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

def create_rsi_chart(
    df
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
            mode="lines",
            name="RSI",
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
        title="RSI(14)",
        height=300,
        yaxis=dict(
            range=[0, 100]
        ),
        margin=dict(
            l=10,
            r=10,
            t=45,
            b=10,
        ),
    )

    return fig


# ============================================================
# MACD CHART
# ============================================================

def create_macd_chart(
    df
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
            mode="lines",
            name="MACD",
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
            mode="lines",
            name="Signal",
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
        title="MACD (12,26,9)",
        height=320,
        margin=dict(
            l=10,
            r=10,
            t=45,
            b=10,
        ),
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

    if not api_key.strip():

        st.error(
            "⚠️ Vui lòng nhập Gemini API Key."
        )

        st.stop()

    # ========================================================
    # 1. THỬ BINANCE REST
    # ========================================================

    with st.spinner(
        "1/5: Đang kiểm tra Binance Futures..."
    ):

        rest_df, rest_error = (
            get_binance_rest_klines(
                SYMBOL,
                interval,
                limit,
            )
        )

    source_label = ""

    # ========================================================
    # NẾU REST CHẠY
    # ========================================================

    if rest_df is not None:

        df = rest_df

        source_label = (
            "Binance Futures REST API"
        )

        st.success(
            "✅ Đang dùng Binance Futures REST."
        )

    else:

        # ====================================================
        # 2. BINANCE DATA ARCHIVE
        # ====================================================

        st.warning(
            "⚠️ Binance Futures REST không truy cập được. "
            "Đang chuyển sang Binance Public Data Archive..."
        )

        with st.spinner(
            "2/5: Đang tải dữ liệu nến CLUSDT từ Binance Archive..."
        ):

            archive_df, archive_source = (
                get_binance_archive_klines(
                    SYMBOL,
                    interval,
                    limit,
                )
            )

        if archive_df is None:

            st.error(
                "❌ Không lấy được dữ liệu CLUSDT từ Binance."
            )

            with st.expander(
                "Chi tiết lỗi REST"
            ):
                st.code(
                    str(rest_error)
                )

            st.warning(
                "App sẽ không tự chuyển sang CL=F. "
                "Bạn đang yêu cầu phân tích Binance CLUSDT, "
                "nên tránh trộn dữ liệu NYMEX/Yahoo vào tín hiệu Binance."
            )

            st.stop()

        df = archive_df

        source_label = archive_source

        st.success(
            "✅ Đang dùng Binance Public Data Archive."
        )

    # ========================================================
    # TECHNICAL
    # ========================================================

    with st.spinner(
        "3/5: Đang tính RSI, MACD và EMA..."
    ):

        df = add_indicators(
            df
        )

        if len(df) < 30:

            st.error(
                "Không đủ dữ liệu để tính chỉ báo."
            )

            st.stop()

        technical = technical_analysis(
            df
        )

    # ========================================================
    # FUNDING + OI
    # ========================================================

    with st.spinner(
        "4/5: Đang lấy Funding Rate và Open Interest..."
    ):

        funding_rate = (
            get_funding_rate(
                SYMBOL
            )
        )

        open_interest = (
            get_open_interest(
                SYMBOL
            )
        )

        mark_data = (
            get_binance_mark_price(
                SYMBOL
            )
        )

    # ========================================================
    # NEWS
    # ========================================================

    news = get_news()

    # ========================================================
    # HEADER STATUS
    # ========================================================

    st.subheader(
        "📊 BINANCE CLUSDT"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

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
            if technical["macd"]
            > technical["signal"]
            else "🔴 Giảm"
        )
    )

    c4.metric(
        "Xu hướng",
        technical["direction"]
    )

    # ========================================================
    # FUTURES
    # ========================================================

    st.subheader(
        "📡 Futures Data"
    )

    f1, f2, f3 = (
        st.columns(3)
    )

    if funding_rate is not None:

        f1.metric(
            "Funding Rate",
            f"{funding_rate * 100:.4f}%",
        )

    else:

        f1.metric(
            "Funding Rate",
            "N/A",
        )

    if open_interest is not None:

        f2.metric(
            "Open Interest",
            f"{open_interest:,.2f}",
        )

    else:

        f2.metric(
            "Open Interest",
            "N/A",
        )

    if mark_data:

        f3.metric(
            "Mark Price",
            f"${mark_data['mark_price']:.2f}",
        )

    else:

        f3.metric(
            "Mark Price",
            "N/A",
        )

    # ========================================================
    # NGUỒN DỮ LIỆU
    # ========================================================

    st.info(
        f"📡 Nguồn dữ liệu nến: {source_label}"
    )

    # ========================================================
    # CANDLESTICK
    # ========================================================

    st.subheader(
        "🕯️ Biểu đồ nến CLUSDT"
    )

    candle_fig = create_candle_chart(
        df
    )

    st.plotly_chart(
        candle_fig,
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

    rsi_fig = create_rsi_chart(
        df
    )

    st.plotly_chart(
        rsi_fig,
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

    macd_fig = create_macd_chart(
        df
    )

    st.plotly_chart(
        macd_fig,
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

    for reason in technical["reasons"]:

        st.write(
            f"• {reason}"
        )

    # ========================================================
    # GEMINI
    # ========================================================

    with st.spinner(
        "5/5: Gemini đang phân tích CLUSDT..."
    ):

        analysis = analyze_with_gemini(
            api_key=api_key.strip(),
            technical=technical,
            funding_rate=funding_rate,
            open_interest=open_interest,
            news=news,
            data_source=source_label,
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

        for index, item in enumerate(
            news,
            start=1,
        ):

            st.write(
                f"**{index}.** {item}"
            )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    now = datetime.now(
        timezone.utc
    )

    st.caption(
        f"Symbol: {SYMBOL} | "
        f"Khung: {interval} | "
        f"Cập nhật: "
        f"{now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    st.caption(
        "⚠️ Dữ liệu và tín hiệu AI chỉ mang tính tham khảo, "
        "không phải lời khuyên đầu tư."
    )
