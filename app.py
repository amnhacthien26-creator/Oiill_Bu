import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
from ddgs import DDGS
from google import genai


# ============================================================
# CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="AI Phân tích Dầu TradingView",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ AI Phân tích Giá Dầu WTI")
st.markdown(
    """
    Biểu đồ giá dầu trực tiếp từ **TradingView**,
    kết hợp RSI, MACD, EMA, tin tức vĩ mô và Gemini AI.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Cấu hình")

gemini_api_key = st.sidebar.text_input(
    "🔑 Gemini API Key",
    type="password",
    placeholder="Dán Gemini API Key"
)

st.sidebar.markdown(
    "[👉 Lấy Gemini API Key]"
    "(https://aistudio.google.com/apikey)"
)

tradingview_symbol = st.sidebar.selectbox(
    "🛢️ Mã dầu trên TradingView",
    [
        "TVC:USOIL",
        "NYMEX:CL1!"
    ],
    index=0
)

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

period = st.sidebar.selectbox(
    "📊 Lịch sử phân tích",
    [
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y"
    ],
    index=2
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **TradingView**
    
    Biểu đồ: TradingView
    
    Technical:
    RSI + MACD + EMA
    
    AI:
    Google Gemini
    """
)

st.sidebar.caption(
    "⚠️ Công cụ chỉ mang tính tham khảo, "
    "không phải lời khuyên đầu tư."
)


# ============================================================
# TRADINGVIEW EMBED
# ============================================================

def tradingview_chart(symbol, tv_interval):
    """
    Nhúng TradingView Advanced Chart Widget.

    TradingView cung cấp widget/chart cho việc hiển thị
    thị trường. Dữ liệu biểu đồ do TradingView cung cấp
    trong widget.
    """

    interval_map = {
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "D"
    }

    tv_interval = interval_map.get(
        tv_interval,
        "60"
    )

    html = f"""
    <div class="tradingview-widget-container"
         style="width:100%; height:650px;">
      <div id="tradingview_chart"
           style="width:100%; height:650px;">
      </div>

      <script
        type="text/javascript"
        src="https://s3.tradingview.com/tv.js">
      </script>

      <script type="text/javascript">

      new TradingView.widget({{
          "autosize": true,
          "symbol": "{symbol}",
          "interval": "{tv_interval}",
          "timezone": "Asia/Bangkok",
          "theme": "dark",
          "style": "1",
          "locale": "vi_VN",
          "toolbar_bg": "#111827",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "hide_top_toolbar": false,
          "hide_legend": false,
          "save_image": true,
          "container_id": "tradingview_chart",
          "studies": [
              "RSI@tv-basicstudies",
              "MACD@tv-basicstudies"
          ]
      }});

      </script>
    </div>
    """

    components.html(
        html,
        height=680,
        scrolling=False
    )


# ============================================================
# YAHOO DATA FOR TECHNICAL CALCULATIONS
# ============================================================

def get_market_data(
    period_value,
    interval_value
):

    # TradingView 4h không có interval tương ứng
    # nên dùng 1h rồi resample.
    source_interval = interval_value

    if interval_value == "4h":
        source_interval = "1h"

    try:

        df = yf.download(
            "CL=F",
            period=period_value,
            interval=source_interval,
            auto_adjust=False,
            progress=False
        )

        if df is None or df.empty:
            return None, (
                "Không lấy được dữ liệu WTI."
            )

        # ----------------------------------------------------
        # MultiIndex
        # ----------------------------------------------------

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        df.columns = [
            str(col).lower()
            for col in df.columns
        ]

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        if not all(
            col in df.columns
            for col in required
        ):

            return None, (
                "Yahoo Finance thiếu dữ liệu OHLC."
            )

        df = df[
            required
        ].copy()

        for col in required:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna()

        # ----------------------------------------------------
        # 4H
        # ----------------------------------------------------

        if interval_value == "4h":

            df = (
                df.resample("4h")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                })
                .dropna()
            )

        df = df.tail(300)

        if len(df) < 50:

            return None, (
                "Không đủ dữ liệu để tính chỉ báo."
            )

        df.index = pd.to_datetime(
            df.index,
            utc=True
        )

        return df, None

    except Exception as exc:

        return None, (
            f"Lỗi lấy dữ liệu dầu: {exc}"
        )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

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

    df["rsi"] = df["rsi"].fillna(50)

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
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
            adjust=False
        )
        .mean()
    )

    df["histogram"] = (
        df["macd"] -
        df["signal"]
    )

    return df


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def technical_analysis(df):

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
        "reasons": reasons
    }


# ============================================================
# NEWS
# ============================================================

def get_global_news(
    max_results=15
):

    queries = [
        "WTI crude oil",
        "OPEC oil",
        "oil inventory",
        "oil geopolitics",
        "WTI oil price",
    ]

    news = []

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

                            news.append(
                                title.strip()
                            )

                except Exception:

                    continue

        news = list(
            dict.fromkeys(
                news
            )
        )

        if not news:

            return [
                "Không lấy được tin tức."
            ]

        return news[
            :max_results
        ]

    except Exception as exc:

        return [
            f"Lỗi tin tức: {exc}"
        ]


# ============================================================
# GEMINI
# ============================================================

def analyze_with_gemini(
    api_key,
    symbol,
    technical,
    news
):

    try:

        client = genai.Client(
            api_key=api_key
        )

        news_text = "\n".join(
            f"- {item}"
            for item in news
        )

        prompt = f"""
Bạn là chuyên gia phân tích WTI Crude Oil.

Biểu đồ tham chiếu:
{symbol}

Dữ liệu OHLC dùng cho chỉ báo:

Giá:
${technical['price']:.2f}

RSI:
{technical['rsi']:.2f}

MACD:
{technical['macd']:.4f}

Signal:
{technical['signal']:.4f}

EMA20:
${technical['ema20']:.2f}

EMA50:
${technical['ema50']:.2f}

Điểm kỹ thuật:
{technical['score']}

Xu hướng:
{technical['direction']}

Tin tức:
{news_text}

Hãy phân tích bằng tiếng Việt.

CẤU TRÚC:

🎯 TÍN HIỆU
LONG / SHORT / WAIT

📈 XU HƯỚNG
Tăng / Giảm / Đi ngang

📊 KỸ THUẬT
Phân tích RSI, MACD, EMA20, EMA50.

📰 VĨ MÔ
Phân tích OPEC, cung cầu,
tồn kho và địa chính trị.

💡 KỊCH BẢN
LONG:
Điều kiện thuận lợi.

SHORT:
Điều kiện thuận lợi.

WAIT:
Điều kiện nên đứng ngoài.

⚠️ RỦI RO
Các yếu tố có thể làm nhận định sai.

QUY TẮC:

- Không khẳng định chắc chắn giá.
- Không bịa dữ liệu.
- Không sử dụng HTML.
- Không sử dụng ký tự lạ.
- Markdown đơn giản.
- Đây chỉ là phân tích tham khảo.
"""

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "thinking_level": "low"
            }
        )

        if (
            hasattr(
                interaction,
                "output_text"
            )
            and interaction.output_text
        ):

            return interaction.output_text

        return (
            "Gemini không trả về nội dung."
        )

    except Exception as exc:

        return (
            "❌ Gemini Error\n\n"
            f"{exc}"
        )


# ============================================================
# APP
# ============================================================

if st.button(
    "🚀 PHÂN TÍCH GIÁ DẦU",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    if not gemini_api_key.strip():

        st.error(
            "⚠️ Bạn chưa nhập Gemini API Key."
        )

        st.stop()

    # --------------------------------------------------------
    # TRADINGVIEW CHART
    # --------------------------------------------------------

    st.subheader(
        "📺 TradingView — Giá dầu trực tiếp"
    )

    tradingview_chart(
        tradingview_symbol,
        interval
    )

    # --------------------------------------------------------
    # TECHNICAL DATA
    # --------------------------------------------------------

    with st.spinner(
        "Đang tải dữ liệu dầu để tính chỉ báo..."
    ):

        df, error = get_market_data(
            period,
            interval
        )

    if df is None:

        st.error(
            error
        )

        st.stop()

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    df = add_indicators(
        df
    )

    technical = (
        technical_analysis(
            df
        )
    )

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    with st.spinner(
        "Đang thu thập tin tức dầu mỏ..."
    ):

        news = get_global_news()

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    st.subheader(
        "📊 Tổng quan kỹ thuật"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "💵 Giá WTI",
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

    # --------------------------------------------------------
    # CANDLE CHART
    # --------------------------------------------------------

    st.subheader(
        "🕯️ Biểu đồ nến + EMA"
    )

    chart_df = df.tail(
        200
    ).copy()

    st.line_chart(
        chart_df[
            [
                "close",
                "ema20",
                "ema50"
            ]
        ],
        height=450
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    st.subheader(
        "📈 RSI"
    )

    st.line_chart(
        df[
            ["rsi"]
        ].tail(200),
        height=280
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    st.subheader(
        "📉 MACD"
    )

    st.line_chart(
        df[
            [
                "macd",
                "signal"
            ]
        ].tail(200),
        height=300
    )

    # --------------------------------------------------------
    # TECHNICAL REASONS
    # --------------------------------------------------------

    st.subheader(
        "🧠 Tín hiệu kỹ thuật"
    )

    for reason in technical[
        "reasons"
    ]:

        st.write(
            f"• {reason}"
        )

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Gemini đang phân tích..."
    ):

        analysis = (
            analyze_with_gemini(
                api_key=gemini_api_key.strip(),
                symbol=tradingview_symbol,
                technical=technical,
                news=news
            )
        )

    st.subheader(
        "🤖 Báo cáo AI"
    )

    st.markdown(
        analysis
    )

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown("---")

    st.caption(
        f"Biểu đồ: TradingView — {tradingview_symbol}"
    )

    st.caption(
        "Chỉ báo RSI/MACD/EMA: dữ liệu OHLC lịch sử WTI."
    )

    st.caption(
        "⚠️ Phân tích chỉ mang tính tham khảo, "
        "không phải lời khuyên đầu tư."
    )
