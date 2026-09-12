import streamlit as st
import yfinance as yf
import pandas as pd
from ddgs import DDGS
from google import genai
from google.genai import types


# ============================================================
# CẤU HÌNH GIAO DIỆN
# ============================================================

st.set_page_config(
    page_title="AI Dự báo Giá Dầu Pro",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ Hệ thống AI Dự báo Giá Dầu")
st.markdown(
    "Tự động thu thập tin tức toàn cầu, phân tích RSI + MACD "
    "và dùng Gemini để tổng hợp xu hướng thị trường WTI."
)

# ============================================================
# API KEY
# ============================================================

api_key = st.sidebar.text_input(
    "🔑 Gemini API Key",
    type="password",
    help="Lấy API Key tại https://aistudio.google.com/"
)

st.sidebar.markdown(
    "[👉 Lấy Gemini API Key tại Google AI Studio]"
    "(https://aistudio.google.com/apikey)"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Đây là công cụ tham khảo, không phải lời khuyên đầu tư."
)


# ============================================================
# LẤY DỮ LIỆU GIÁ + RSI + MACD
# ============================================================

def get_technical_data(ticker="CL=F", days=90):
    """
    Lấy giá WTI từ Yahoo Finance.
    Tính RSI 14 và MACD 12/26/9.
    """

    try:
        data = yf.download(
            ticker,
            period=f"{days}d",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data is None or data.empty:
            return None, "Không lấy được dữ liệu giá WTI từ Yahoo Finance."

        # Xử lý MultiIndex của yfinance
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(ticker, axis=1, level=-1)
            except Exception:
                data.columns = data.columns.get_level_values(0)

        data.columns = [str(c) for c in data.columns]

        if "Close" not in data.columns:
            return None, "Dữ liệu Yahoo Finance không có cột Close."

        # Chuyển Close thành dữ liệu số
        close = pd.to_numeric(
            data["Close"],
            errors="coerce"
        )

        data = data.copy()
        data["Close"] = close
        data = data.dropna(subset=["Close"])

        if len(data) < 35:
            return None, "Không đủ dữ liệu để tính RSI/MACD."

        # ====================================================
        # RSI 14
        # ====================================================

        delta = data["Close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(
            window=14,
            min_periods=14
        ).mean()

        avg_loss = loss.rolling(
            window=14,
            min_periods=14
        ).mean()

        rs = avg_gain / avg_loss.replace(0, pd.NA)

        data["RSI"] = 100 - (100 / (1 + rs))
        data["RSI"] = data["RSI"].fillna(50)

        # ====================================================
        # MACD
        # ====================================================

        ema12 = data["Close"].ewm(
            span=12,
            adjust=False
        ).mean()

        ema26 = data["Close"].ewm(
            span=26,
            adjust=False
        ).mean()

        data["MACD"] = ema12 - ema26

        data["Signal"] = data["MACD"].ewm(
            span=9,
            adjust=False
        ).mean()

        data = data.dropna(
            subset=["RSI", "MACD", "Signal"]
        )

        if data.empty:
            return None, "Không đủ dữ liệu sau khi tính RSI/MACD."

        return data.tail(30), None

    except Exception as e:
        return None, f"Lỗi khi tải dữ liệu thị trường: {str(e)}"


# ============================================================
# LẤY TIN TỨC
# ============================================================

def get_global_news(
    query="crude oil OPEC WTI oil price energy",
    max_results=15
):
    """
    Lấy tin tức mới nhất thông qua DuckDuckGo.
    """

    try:
        news = []

        with DDGS() as ddgs:
            results = ddgs.news(
                query=query,
                max_results=max_results
            )

            for item in results:
                title = item.get("title")

                if title:
                    news.append(title.strip())

        if not news:
            return [
                "Không tìm thấy tin tức mới. "
                "Hãy thận trọng khi đánh giá yếu tố vĩ mô."
            ]

        return news

    except Exception as e:
        return [
            "Không thể lấy tin tức từ DuckDuckGo tại thời điểm này.",
            f"Chi tiết kỹ thuật: {str(e)}"
        ]


# ============================================================
# GEMINI AI
# ============================================================

def ai_analyze_market(
    news_list,
    last_price,
    rsi,
    macd,
    signal,
    api_key
):
    """
    Gửi dữ liệu sang Gemini để phân tích.
    """

    try:
        client = genai.Client(
            api_key=api_key
        )

        news_text = "\n".join(
            [f"- {news}" for news in news_list]
        )

        # Trạng thái RSI
        if rsi >= 70:
            rsi_status = "Quá mua"
        elif rsi <= 30:
            rsi_status = "Quá bán"
        else:
            rsi_status = "Trung tính"

        # Trạng thái MACD
        if macd > signal:
            macd_status = (
                "MACD nằm trên đường Signal, "
                "thiên về tăng"
            )
        else:
            macd_status = (
                "MACD nằm dưới đường Signal, "
                "thiên về giảm"
            )

        prompt = f"""
Bạn là chuyên gia phân tích thị trường dầu thô WTI.

Hãy phân tích dữ liệu dưới đây một cách ngắn gọn,
dễ hiểu và có tính thực tế.

=============================
DỮ LIỆU THỊ TRƯỜNG
=============================

Giá WTI hiện tại:
${last_price:.2f}

RSI(14):
{rsi:.2f}

Trạng thái RSI:
{rsi_status}

MACD:
{macd:.4f}

Signal:
{signal:.4f}

Trạng thái MACD:
{macd_status}

=============================
TIN TỨC VĨ MÔ
=============================

{news_text}

=============================
YÊU CẦU
=============================

Hãy trả lời bằng tiếng Việt.

Không cần giải thích quá dài.

Sử dụng đúng cấu trúc:

🎯 XU HƯỚNG DỰ KIẾN
Tăng / Giảm / Đi ngang

📰 TÁC ĐỘNG VĨ MÔ
Tóm tắt các yếu tố cung/cầu, OPEC,
tồn kho, địa chính trị và nhu cầu dầu.

📊 TÍN HIỆU KỸ THUẬT
Nhận xét RSI + MACD.

💡 CHIẾN LƯỢC THAM KHẢO
Nêu vùng cần quan sát và rủi ro.

⚠️ RỦI RO
Nêu các yếu tố có thể khiến dự báo sai.

Không khẳng định chắc chắn giá sẽ tăng hoặc giảm.
"""

        # Model Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1200
            )
        )

        if not response or not response.text:
            return "Gemini không trả về nội dung phân tích."

        return response.text

    except Exception as e:
        return (
            "❌ Không thể kết nối Gemini.\n\n"
            f"Chi tiết lỗi: {str(e)}\n\n"
            "Hãy kiểm tra:\n"
            "1. Gemini API Key có đúng không.\n"
            "2. API Key còn hoạt động không.\n"
            "3. API Gemini đã được bật cho key chưa.\n"
            "4. Kiểm tra quota của API."
        )


# ============================================================
# CHUYỂN GIÁ TRỊ VỀ FLOAT
# ============================================================

def safe_float(value):

    try:
        if hasattr(value, "item"):
            value = value.item()

        return float(value)

    except Exception:
        return float("nan")


# ============================================================
# LUỒNG CHÍNH
# ============================================================

if st.button(
    "🚀 Bắt đầu Quét & Phân tích",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # KIỂM TRA API KEY
    # ========================================================

    if not api_key or not api_key.strip():

        st.error(
            "⚠️ Bạn chưa nhập Gemini API Key ở menu bên trái."
        )

        st.stop()

    # ========================================================
    # BƯỚC 1: DỮ LIỆU KỸ THUẬT
    # ========================================================

    with st.spinner(
        "1/3: Đang tải dữ liệu giá WTI và tính RSI/MACD..."
    ):

        df, error_message = get_technical_data()

    if df is None:

        st.error(error_message)
        st.stop()

    try:

        last_price = safe_float(
            df["Close"].iloc[-1]
        )

        last_rsi = safe_float(
            df["RSI"].iloc[-1]
        )

        last_macd = safe_float(
            df["MACD"].iloc[-1]
        )

        last_signal = safe_float(
            df["Signal"].iloc[-1]
        )

        if any(
            pd.isna(x)
            for x in [
                last_price,
                last_rsi,
                last_macd,
                last_signal
            ]
        ):

            st.error(
                "Dữ liệu kỹ thuật bị thiếu hoặc không hợp lệ."
            )

            st.stop()

    except Exception as e:

        st.error(
            f"Không thể đọc dữ liệu kỹ thuật: {str(e)}"
        )

        st.stop()

    # ========================================================
    # BƯỚC 2: TIN TỨC
    # ========================================================

    with st.spinner(
        "2/3: Đang quét tin tức dầu mỏ và địa chính trị..."
    ):

        news = get_global_news()

    # ========================================================
    # HIỂN THỊ THÔNG SỐ
    # ========================================================

    st.subheader("📊 Trạng thái thị trường")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Giá WTI",
        f"${last_price:.2f}"
    )

    col2.metric(
        "RSI",
        f"{last_rsi:.2f}"
    )

    if last_rsi >= 70:

        rsi_label = "🔴 Quá mua"

    elif last_rsi <= 30:

        rsi_label = "🟢 Quá bán"

    else:

        rsi_label = "🟡 Trung tính"

    col3.metric(
        "RSI trạng thái",
        rsi_label
    )

    if last_macd > last_signal:

        macd_status = "🟢 MACD tích cực"

    else:

        macd_status = "🔴 MACD tiêu cực"

    col4.metric(
        "MACD",
        macd_status
    )

    # ========================================================
    # BIỂU ĐỒ
    # ========================================================

    st.subheader("📈 Biểu đồ WTI")

    chart_data = df[["Close"]].copy()

    st.line_chart(
        chart_data,
        height=350
    )

    # ========================================================
    # BƯỚC 3: AI
    # ========================================================

    with st.spinner(
        "3/3: Gemini đang phân tích dữ liệu thị trường..."
    ):

        analysis = ai_analyze_market(
            news,
            last_price,
            last_rsi,
            last_macd,
            last_signal,
            api_key.strip()
        )

    # ========================================================
    # KẾT QUẢ AI
    # ========================================================

    st.subheader("🤖 Báo cáo phân tích của AI")

    st.markdown(analysis)

    # ========================================================
    # TIN TỨC
    # ========================================================

    st.subheader("📰 Dữ liệu tin tức đầu vào")

    with st.expander(
        "Xem các bản tin đã thu thập"
    ):

        if news:

            for i, item in enumerate(
                news,
                start=1
            ):

                st.write(
                    f"**{i}.** {item}"
                )

        else:

            st.write(
                "Không có tin tức."
            )

    # ========================================================
    # FOOTER
    # ========================================================

    st.caption(
        "Nguồn giá: Yahoo Finance / WTI Futures (CL=F). "
        "Nguồn tin: DuckDuckGo News. "
        "Phân tích AI: Google Gemini."
    )
