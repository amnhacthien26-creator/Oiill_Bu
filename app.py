import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
from ddgs import DDGS
from google import genai


# ============================================================
# CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="AI Phân tích Dầu Binance",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ AI Phân tích Giá Dầu Binance")
st.markdown(
    """
    Phân tích **WTI Crude Oil trên Binance Futures (CLUSDT)**
    bằng dữ liệu giá, RSI, MACD, Volume, Funding Rate,
    Open Interest và tin tức vĩ mô.
    """
)

# Binance public Futures API
BINANCE_BASE_URL = "https://fapi.binance.com"

# Hợp đồng dầu WTI trên Binance
SYMBOL = "CLUSDT"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Cấu hình")

api_key = st.sidebar.text_input(
    "🔑 Gemini API Key",
    type="password",
    placeholder="Dán Gemini API Key vào đây"
)

st.sidebar.markdown(
    "[👉 Lấy Gemini API Key]"
    "(https://aistudio.google.com/apikey)"
)

st.sidebar.markdown("---")

interval = st.sidebar.selectbox(
    "⏱️ Khung thời gian",
    [
        "5m",
        "15m",
        "30m",
        "1h",
        "4h",
        "1d"
    ],
    index=3
)

kline_limit = st.sidebar.selectbox(
    "📊 Số nến phân tích",
    [100, 200, 300, 500],
    index=1
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Binance Futures**

    Symbol: CLUSDT  
    Tài sản: WTI Crude Oil  
    Giao dịch: 24/7

    ⚠️ Dữ liệu và AI chỉ mang tính
    chất tham khảo, không phải lời
    khuyên đầu tư.
    """
)


# ============================================================
# HÀM GỌI BINANCE API
# ============================================================

def binance_get(endpoint, params=None, timeout=15):
    """
    Gọi Binance Futures Public API.
    """

    try:
        url = BINANCE_BASE_URL + endpoint

        response = requests.get(
            url,
            params=params or {},
            timeout=timeout
        )

        response.raise_for_status()

        return response.json(), None

    except requests.exceptions.RequestException as e:

        return (
            None,
            f"Lỗi kết nối Binance: {e}"
        )

    except Exception as e:

        return (
            None,
            f"Lỗi Binance API: {e}"
        )


# ============================================================
# LẤY CANDLESTICK
# ============================================================

def get_binance_klines(
    symbol=SYMBOL,
    interval="1h",
    limit=200
):
    """
    Lấy dữ liệu nến Binance Futures.
    """

    data, error = binance_get(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )

    if error:
        return None, error

    if not data:
        return None, "Binance không trả về dữ liệu nến."

    try:

        columns = [
            "OpenTime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "CloseTime",
            "QuoteVolume",
            "Trades",
            "TakerBuyBaseVolume",
            "TakerBuyQuoteVolume",
            "Ignore"
        ]

        df = pd.DataFrame(
            data,
            columns=columns
        )

        # ----------------------------------------------------
        # Chuyển kiểu dữ liệu
        # ----------------------------------------------------

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "QuoteVolume",
            "Trades",
            "TakerBuyBaseVolume",
            "TakerBuyQuoteVolume"
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df["OpenTime"] = pd.to_datetime(
            df["OpenTime"],
            unit="ms",
            utc=True
        )

        df["CloseTime"] = pd.to_datetime(
            df["CloseTime"],
            unit="ms",
            utc=True
        )

        # ====================================================
        # RSI
        # ====================================================

        delta = df["Close"].diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        avg_gain = gain.rolling(
            window=14,
            min_periods=14
        ).mean()

        avg_loss = loss.rolling(
            window=14,
            min_periods=14
        ).mean()

        avg_loss = avg_loss.replace(
            0,
            pd.NA
        )

        rs = avg_gain / avg_loss

        df["RSI"] = (
            100 -
            (
                100 /
                (1 + rs)
            )
        )

        df["RSI"] = df["RSI"].fillna(50)

        # ====================================================
        # MACD
        # ====================================================

        ema12 = df["Close"].ewm(
            span=12,
            adjust=False
        ).mean()

        ema26 = df["Close"].ewm(
            span=26,
            adjust=False
        ).mean()

        df["MACD"] = (
            ema12 -
            ema26
        )

        df["Signal"] = (
            df["MACD"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        df["MACD_Histogram"] = (
            df["MACD"] -
            df["Signal"]
        )

        # ====================================================
        # EMA
        # ====================================================

        df["EMA20"] = (
            df["Close"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
        )

        df["EMA50"] = (
            df["Close"]
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
        )

        df = df.dropna(
            subset=[
                "Close",
                "RSI",
                "MACD",
                "Signal"
            ]
        )

        return df, None

    except Exception as e:

        return (
            None,
            f"Lỗi xử lý dữ liệu nến: {e}"
        )


# ============================================================
# GIÁ HIỆN TẠI
# ============================================================

def get_current_price(
    symbol=SYMBOL
):
    """
    Lấy giá hiện tại của CLUSDT.
    """

    data, error = binance_get(
        "/fapi/v1/ticker/price",
        {
            "symbol": symbol
        }
    )

    if error:
        return None, error

    try:

        return float(
            data["price"]
        ), None

    except Exception:

        return (
            None,
            "Không đọc được giá hiện tại."
        )


# ============================================================
# FUNDING RATE
# ============================================================

def get_funding_rate(
    symbol=SYMBOL
):
    """
    Lấy funding rate gần nhất.
    """

    data, error = binance_get(
        "/fapi/v1/fundingRate",
        {
            "symbol": symbol,
            "limit": 1
        }
    )

    if error:
        return None, error

    if not data:
        return None, "Không có Funding Rate."

    try:

        funding = float(
            data[-1]["fundingRate"]
        )

        funding_time = pd.to_datetime(
            data[-1]["fundingTime"],
            unit="ms",
            utc=True
        )

        return {
            "rate": funding,
            "time": funding_time
        }, None

    except Exception as e:

        return (
            None,
            f"Lỗi Funding Rate: {e}"
        )


# ============================================================
# OPEN INTEREST
# ============================================================

def get_open_interest(
    symbol=SYMBOL
):
    """
    Lấy Open Interest hiện tại.
    """

    data, error = binance_get(
        "/fapi/v1/openInterest",
        {
            "symbol": symbol
        }
    )

    if error:
        return None, error

    try:

        return float(
            data["openInterest"]
        ), None

    except Exception:

        return (
            None,
            "Không đọc được Open Interest."
        )


# ============================================================
# LONG / SHORT RATIO
# ============================================================

def get_long_short_ratio(
    symbol=SYMBOL,
    period="1h"
):
    """
    Lấy tỷ lệ Long/Short của Binance Futures.

    Nếu Binance không trả dữ liệu cho symbol này,
    app vẫn tiếp tục hoạt động.
    """

    data, error = binance_get(
        "/futures/data/globalLongShortAccountRatio",
        {
            "symbol": symbol,
            "period": period,
            "limit": 1
        }
    )

    if error:
        return None, error

    if not data:
        return None, "Không có dữ liệu Long/Short."

    try:

        item = data[-1]

        return {
            "ratio": float(
                item["longShortRatio"]
            ),
            "long": float(
                item["longAccount"]
            ) * 100,
            "short": float(
                item["shortAccount"]
            ) * 100
        }, None

    except Exception as e:

        return (
            None,
            f"Lỗi Long/Short: {e}"
        )


# ============================================================
# TIN TỨC
# ============================================================

def get_global_news(
    max_results=15
):
    """
    Tìm tin tức dầu mỏ, OPEC, địa chính trị.
    """

    queries = [
        "WTI crude oil price",
        "OPEC oil production",
        "crude oil inventory",
        "oil geopolitics",
        "US crude oil demand"
    ]

    all_news = []

    try:

        with DDGS() as ddgs:

            for query in queries:

                try:

                    results = ddgs.news(
                        query=query,
                        max_results=4
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

        # Loại bỏ tin trùng
        unique_news = list(
            dict.fromkeys(
                all_news
            )
        )

        if not unique_news:

            return [
                "Không lấy được tin tức mới."
            ]

        return unique_news[:max_results]

    except Exception as e:

        return [
            "Không thể lấy tin tức.",
            f"Lỗi nguồn tin: {e}"
        ]


# ============================================================
# PHÂN TÍCH KỸ THUẬT
# ============================================================

def technical_signal(
    rsi,
    macd,
    signal,
    close,
    ema20,
    ema50
):

    score = 0
    reasons = []

    # RSI
    if rsi < 30:

        score += 2

        reasons.append(
            "RSI quá bán → có khả năng hồi phục."
        )

    elif rsi > 70:

        score -= 2

        reasons.append(
            "RSI quá mua → có nguy cơ điều chỉnh."
        )

    elif rsi >= 50:

        score += 1

        reasons.append(
            "RSI trên 50 → động lượng tăng tốt hơn."
        )

    else:

        score -= 1

        reasons.append(
            "RSI dưới 50 → động lượng yếu."
        )

    # MACD
    if macd > signal:

        score += 2

        reasons.append(
            "MACD trên Signal → tín hiệu tăng."
        )

    else:

        score -= 2

        reasons.append(
            "MACD dưới Signal → tín hiệu giảm."
        )

    # EMA
    if close > ema20:

        score += 1

        reasons.append(
            "Giá trên EMA20."
        )

    else:

        score -= 1

        reasons.append(
            "Giá dưới EMA20."
        )

    if close > ema50:

        score += 1

        reasons.append(
            "Giá trên EMA50."
        )

    else:

        score -= 1

        reasons.append(
            "Giá dưới EMA50."
        )

    # Tổng hợp
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

    return direction, score, reasons


# ============================================================
# GEMINI AI
# ============================================================

def ai_analyze_market(
    news,
    price,
    rsi,
    macd,
    signal,
    volume,
    funding_rate,
    open_interest,
    long_short,
    technical_direction,
    technical_score,
    api_key
):

    try:

        client = genai.Client(
            api_key=api_key
        )

        news_text = "\n".join(
            [
                f"- {item}"
                for item in news
            ]
        )

        # ====================================================
        # Funding
        # ====================================================

        funding_percent = (
            funding_rate * 100
        )

        if funding_rate > 0:

            funding_status = (
                "Funding dương → Long đang trả phí cho Short."
            )

        elif funding_rate < 0:

            funding_status = (
                "Funding âm → Short đang trả phí cho Long."
            )

        else:

            funding_status = (
                "Funding gần trung tính."
            )

        # ====================================================
        # Long Short
        # ====================================================

        if long_short:

            ls_ratio = (
                long_short["ratio"]
            )

            ls_long = (
                long_short["long"]
            )

            ls_short = (
                long_short["short"]
            )

            if ls_ratio > 1:

                ls_status = (
                    "Tỷ lệ tài khoản Long cao hơn Short."
                )

            elif ls_ratio < 1:

                ls_status = (
                    "Tỷ lệ tài khoản Short cao hơn Long."
                )

            else:

                ls_status = (
                    "Long và Short khá cân bằng."
                )

        else:

            ls_ratio = 0
            ls_long = 0
            ls_short = 0

            ls_status = (
                "Không có dữ liệu Long/Short."
            )

        # ====================================================
        # RSI
        # ====================================================

        if rsi >= 70:

            rsi_status = "QUÁ MUA"

        elif rsi <= 30:

            rsi_status = "QUÁ BÁN"

        elif rsi >= 50:

            rsi_status = "TÍCH CỰC"

        else:

            rsi_status = "TIÊU CỰC"

        # ====================================================
        # MACD
        # ====================================================

        if macd > signal:

            macd_status = "TĂNG"

        else:

            macd_status = "GIẢM"

        # ====================================================
        # PROMPT
        # ====================================================

        prompt = f"""
Bạn là chuyên gia phân tích WTI Crude Oil
trên Binance Futures.

Phân tích toàn bộ dữ liệu dưới đây.

====================================================
THỊ TRƯỜNG
====================================================

Symbol:
CLUSDT

Giá hiện tại:
${price:.2f}

Khung thời gian:
{interval}

Volume:
{volume:,.2f}

====================================================
TECHNICAL
====================================================

RSI:
{rsi:.2f}

Trạng thái RSI:
{rsi_status}

MACD:
{macd:.4f}

Signal:
{signal:.4f}

MACD:
{macd_status}

Tín hiệu tổng hợp kỹ thuật:
{technical_direction}

Điểm kỹ thuật:
{technical_score}

====================================================
BINANCE FUTURES
====================================================

Funding Rate:
{funding_percent:.4f}%

Trạng thái Funding:
{funding_status}

Open Interest:
{open_interest:,.2f}

Long/Short Ratio:
{ls_ratio:.4f}

Long Account:
{ls_long:.2f}%

Short Account:
{ls_short:.2f}%

Nhận định Long/Short:
{ls_status}

====================================================
TIN TỨC
====================================================

{news_text}

====================================================
YÊU CẦU
====================================================

Hãy phân tích bằng tiếng Việt.

Không chỉ dựa vào RSI/MACD.
Hãy kết hợp:

- Giá
- RSI
- MACD
- Volume
- EMA
- Funding Rate
- Open Interest
- Long/Short
- Tin tức
- OPEC
- Tồn kho dầu
- Cung cầu
- Địa chính trị

Trả lời đúng cấu trúc:

🎯 TÍN HIỆU HIỆN TẠI

LONG / SHORT / WAIT

📈 XU HƯỚNG

Tăng / Giảm / Đi ngang

🧠 PHÂN TÍCH

Giải thích ngắn gọn nhưng cụ thể.

📊 TECHNICAL

Phân tích RSI + MACD + Volume + EMA.

💰 FUTURES SENTIMENT

Phân tích Funding + Open Interest + Long/Short.

📰 TÁC ĐỘNG VĨ MÔ

Phân tích OPEC, tồn kho, cung cầu,
địa chính trị và các tin quan trọng.

💡 KỊCH BẢN GIAO DỊCH

Nêu:

- Điều kiện thuận lợi cho LONG
- Điều kiện thuận lợi cho SHORT
- Khi nào nên WAIT

⚠️ RỦI RO

Nêu các yếu tố có thể khiến nhận định sai.

Không được khẳng định chắc chắn thị trường
sẽ tăng hay giảm.

Đây chỉ là công cụ phân tích tham khảo,
không phải lời khuyên đầu tư.
"""

        # ====================================================
        # GEMINI INTERACTIONS API
        # ====================================================

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
        ):

            if interaction.output_text:

                return interaction.output_text

        # Fallback
        if hasattr(
            interaction,
            "steps"
        ):

            for step in reversed(
                interaction.steps
            ):

                if hasattr(
                    step,
                    "content"
                ):

                    content = step.content

                    if isinstance(
                        content,
                        list
                    ):

                        for item in content:

                            if (
                                hasattr(
                                    item,
                                    "text"
                                )
                                and item.text
                            ):

                                return item.text

        return (
            "Gemini không trả về nội dung."
        )

    except Exception as e:

        return (
            "❌ GEMINI ERROR\n\n"
            f"{str(e)}"
        )


# ============================================================
# NÚT PHÂN TÍCH
# ============================================================

if st.button(
    "🚀 PHÂN TÍCH CLUSDT",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # KIỂM TRA GEMINI KEY
    # ========================================================

    if not api_key:

        st.error(
            "⚠️ Bạn chưa nhập Gemini API Key."
        )

        st.stop()

    api_key = api_key.strip()

    # ========================================================
    # 1. BINANCE KLINE
    # ========================================================

    with st.spinner(
        "1/5: Đang tải dữ liệu Binance CLUSDT..."
    ):

        df, error = get_binance_klines(
            SYMBOL,
            interval,
            kline_limit
        )

    if df is None:

        st.error(error)
        st.stop()

    # ========================================================
    # LẤY GIÁ
    # ========================================================

    last = df.iloc[-1]

    price = float(
        last["Close"]
    )

    rsi = float(
        last["RSI"]
    )

    macd = float(
        last["MACD"]
    )

    signal = float(
        last["Signal"]
    )

    volume = float(
        last["Volume"]
    )

    ema20 = float(
        last["EMA20"]
    )

    ema50 = float(
        last["EMA50"]
    )

    # ========================================================
    # TECHNICAL SIGNAL
    # ========================================================

    (
        technical_direction,
        technical_score,
        technical_reasons
    ) = technical_signal(
        rsi,
        macd,
        signal,
        price,
        ema20,
        ema50
    )

    # ========================================================
    # 2. FUNDING
    # ========================================================

    with st.spinner(
        "2/5: Đang lấy Funding Rate..."
    ):

        funding_data, _ = (
            get_funding_rate()
        )

    if funding_data:

        funding_rate = float(
            funding_data["rate"]
        )

    else:

        funding_rate = 0.0

    # ========================================================
    # 3. OPEN INTEREST
    # ========================================================

    with st.spinner(
        "3/5: Đang lấy Open Interest..."
    ):

        open_interest, _ = (
            get_open_interest()
        )

    if open_interest is None:

        open_interest = 0.0

    # ========================================================
    # LONG SHORT
    # ========================================================

    long_short, _ = (
        get_long_short_ratio(
            SYMBOL,
            "1h"
        )
    )

    # ========================================================
    # 4. NEWS
    # ========================================================

    with st.spinner(
        "4/5: Đang quét tin tức dầu mỏ..."
    ):

        news = get_global_news()

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.subheader(
        "📊 BINANCE CLUSDT"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "💵 Giá",
        f"${price:.2f}"
    )

    col2.metric(
        "RSI",
        f"{rsi:.2f}"
    )

    col3.metric(
        "MACD",
        "🟢 Tăng"
        if macd > signal
        else "🔴 Giảm"
    )

    col4.metric(
        "Kỹ thuật",
        technical_direction
    )

    # ========================================================
    # FUTURES DATA
    # ========================================================

    st.subheader(
        "📡 Binance Futures"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Funding Rate",
        f"{funding_rate * 100:.4f}%"
    )

    col2.metric(
        "Open Interest",
        f"{open_interest:,.2f}"
    )

    if long_short:

        col3.metric(
            "Long",
            f"{long_short['long']:.2f}%"
        )

        col4.metric(
            "Short",
            f"{long_short['short']:.2f}%"
        )

    else:

        col3.metric(
            "Long",
            "N/A"
        )

        col4.metric(
            "Short",
            "N/A"
        )

    # ========================================================
    # BIỂU ĐỒ GIÁ
    # ========================================================

    st.subheader(
        f"📈 CLUSDT - {interval}"
    )

    chart = df[
        [
            "Close",
            "EMA20",
            "EMA50"
        ]
    ].copy()

    chart.columns = [
        "Giá",
        "EMA20",
        "EMA50"
    ]

    st.line_chart(
        chart,
        height=400
    )

    # ========================================================
    # RSI
    # ========================================================

    st.subheader(
        "📊 RSI"
    )

    st.line_chart(
        df[
            ["RSI"]
        ],
        height=250
    )

    # ========================================================
    # MACD
    # ========================================================

    st.subheader(
        "📉 MACD"
    )

    macd_chart = df[
        [
            "MACD",
            "Signal"
        ]
    ].copy()

    st.line_chart(
        macd_chart,
        height=250
    )

    # ========================================================
    # TECHNICAL REASONS
    # ========================================================

    st.subheader(
        "🧠 Phân tích kỹ thuật tự động"
    )

    for reason in technical_reasons:

        st.write(
            "• " + reason
        )

    # ========================================================
    # 5. GEMINI
    # ========================================================

    with st.spinner(
        "5/5: Gemini đang tổng hợp toàn bộ dữ liệu..."
    ):

        analysis = ai_analyze_market(
            news=news,
            price=price,
            rsi=rsi,
            macd=macd,
            signal=signal,
            volume=volume,
            funding_rate=funding_rate,
            open_interest=open_interest,
            long_short=long_short,
            technical_direction=technical_direction,
            technical_score=technical_score,
            api_key=api_key
        )

    # ========================================================
    # AI RESULT
    # ========================================================

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
            start=1
        ):

            st.write(
                f"**{i}.** {item}"
            )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    now = datetime.now(
        timezone.utc
    )

    st.caption(
        f"""
        Binance Futures: {SYMBOL} |
        Khung: {interval} |
        Cập nhật: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}

        ⚠️ Công cụ phân tích tham khảo, không phải lời khuyên đầu tư.
        """
    )
