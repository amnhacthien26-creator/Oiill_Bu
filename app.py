import streamlit as st
import yfinance as yf
import pandas as pd
from ddgs import DDGS
from google import genai


# ============================================================
# CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="AI Dự báo Giá Dầu Pro",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ AI Dự báo Giá Dầu WTI")
st.markdown(
    "Phân tích giá dầu WTI bằng dữ liệu kỹ thuật RSI + MACD "
    "kết hợp tin tức vĩ mô và Gemini AI."
)

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
    "[👉 Lấy Gemini API Key tại Google AI Studio]"
    "(https://aistudio.google.com/apikey)"
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Công cụ chỉ mang tính chất tham khảo, "
    "không phải lời khuyên đầu tư."
)


# ============================================================
# LẤY DỮ LIỆU GIÁ
# ============================================================

def get_technical_data(
    ticker="CL=F",
    days=90
):
    """
    Lấy dữ liệu WTI từ Yahoo Finance
    và tính RSI + MACD.
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
            return None, "Không lấy được dữ liệu WTI từ Yahoo Finance."

        # ----------------------------------------------------
        # Xử lý MultiIndex của yfinance
        # ----------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):

            try:

                data = data.xs(
                    ticker,
                    axis=1,
                    level=-1
                )

            except Exception:

                data.columns = (
                    data.columns
                    .get_level_values(0)
                )

        data.columns = [
            str(column)
            for column in data.columns
        ]

        # ----------------------------------------------------
        # Kiểm tra Close
        # ----------------------------------------------------

        if "Close" not in data.columns:

            return (
                None,
                "Dữ liệu Yahoo Finance không có cột Close."
            )

        data["Close"] = pd.to_numeric(
            data["Close"],
            errors="coerce"
        )

        data = data.dropna(
            subset=["Close"]
        )

        if len(data) < 35:

            return (
                None,
                "Không đủ dữ liệu để tính RSI và MACD."
            )

        # ====================================================
        # RSI 14
        # ====================================================

        delta = data["Close"].diff()

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

        # Tránh chia cho 0
        avg_loss = avg_loss.replace(
            0,
            pd.NA
        )

        rs = avg_gain / avg_loss

        data["RSI"] = (
            100 -
            (
                100 /
                (1 + rs)
            )
        )

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

        data["MACD"] = (
            ema12 -
            ema26
        )

        data["Signal"] = (
            data["MACD"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        data = data.dropna(
            subset=[
                "RSI",
                "MACD",
                "Signal"
            ]
        )

        if data.empty:

            return (
                None,
                "Không đủ dữ liệu sau khi tính RSI/MACD."
            )

        return data.tail(30), None

    except Exception as e:

        return (
            None,
            f"Lỗi khi lấy dữ liệu thị trường: {e}"
        )


# ============================================================
# LẤY TIN TỨC
# ============================================================

def get_global_news(
    query="crude oil OPEC WTI oil price energy",
    max_results=15
):
    """
    Thu thập tin tức dầu mỏ mới nhất.
    """

    try:

        news = []

        with DDGS() as ddgs:

            results = ddgs.news(
                query=query,
                max_results=max_results
            )

            for item in results:

                title = item.get(
                    "title"
                )

                if title:

                    news.append(
                        title.strip()
                    )

        if not news:

            return [
                "Không tìm thấy tin tức mới."
            ]

        return news

    except Exception as e:

        return [
            "Không thể lấy tin tức tại thời điểm này.",
            f"Lỗi nguồn tin: {e}"
        ]


# ============================================================
# CHUYỂN GIÁ TRỊ THÀNH FLOAT
# ============================================================

def safe_float(value):

    try:

        if hasattr(
            value,
            "item"
        ):

            value = value.item()

        return float(value)

    except Exception:

        return float("nan")


# ============================================================
# GEMINI INTERACTIONS API
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
    Phân tích thị trường bằng Gemini 3.6 Flash
    thông qua Interactions API.
    """

    try:

        # ----------------------------------------------------
        # Tạo Gemini Client
        # ----------------------------------------------------

        client = genai.Client(
            api_key=api_key
        )

        # ----------------------------------------------------
        # Tin tức
        # ----------------------------------------------------

        news_text = "\n".join(
            [
                f"- {news}"
                for news in news_list
            ]
        )

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        if rsi >= 70:

            rsi_status = (
                "QUÁ MUA - thị trường có thể đang tăng nóng"
            )

        elif rsi <= 30:

            rsi_status = (
                "QUÁ BÁN - thị trường có thể đang giảm mạnh"
            )

        else:

            rsi_status = (
                "TRUNG TÍNH"
            )

        # ----------------------------------------------------
        # MACD
        # ----------------------------------------------------

        if macd > signal:

            macd_status = (
                "TÍCH CỰC - MACD nằm trên Signal"
            )

        elif macd < signal:

            macd_status = (
                "TIÊU CỰC - MACD nằm dưới Signal"
            )

        else:

            macd_status = (
                "TRUNG TÍNH"
            )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = f"""
Bạn là chuyên gia phân tích thị trường dầu thô WTI.

Hãy phân tích dữ liệu hiện tại và đưa ra nhận định
ngắn gọn, rõ ràng bằng tiếng Việt.

==================================================
DỮ LIỆU GIÁ
==================================================

Giá WTI hiện tại:
${last_price:.2f}

==================================================
CHỈ BÁO KỸ THUẬT
==================================================

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

==================================================
TIN TỨC VĨ MÔ
==================================================

{news_text}

==================================================
YÊU CẦU
==================================================

Phân tích các yếu tố:

1. Xu hướng giá hiện tại.
2. Cung và cầu dầu.
3. OPEC.
4. Tồn kho dầu Mỹ nếu có.
5. Địa chính trị.
6. Nhu cầu dầu toàn cầu.
7. RSI.
8. MACD.
9. Rủi ro thị trường.

Trả lời đúng cấu trúc sau:

🎯 XU HƯỚNG DỰ KIẾN

Tăng / Giảm / Đi ngang

Giải thích ngắn gọn.

📰 TÁC ĐỘNG VĨ MÔ

Nêu các yếu tố tích cực và tiêu cực
ảnh hưởng đến giá dầu.

📊 TÍN HIỆU KỸ THUẬT

Phân tích RSI + MACD.

💡 CHIẾN LƯỢC THAM KHẢO

Nêu vùng cần quan sát,
điều kiện có thể cân nhắc mua,
điều kiện có thể cân nhắc bán.

⚠️ RỦI RO

Nêu các yếu tố có thể làm dự báo sai.

Không được khẳng định chắc chắn giá sẽ tăng
hoặc giảm.

Đây chỉ là phân tích tham khảo,
không phải lời khuyên đầu tư.
"""

        # ----------------------------------------------------
        # GỌI GEMINI INTERACTIONS API
        # ----------------------------------------------------

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "thinking_level": "low"
            }
        )

        # ----------------------------------------------------
        # KẾT QUẢ
        # ----------------------------------------------------

        if not interaction:

            return (
                "Gemini không trả về dữ liệu."
            )

        if hasattr(
            interaction,
            "output_text"
        ):

            if interaction.output_text:

                return (
                    interaction.output_text
                )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if hasattr(
            interaction,
            "steps"
        ):

            for step in reversed(
                interaction.steps
            ):

                if not hasattr(
                    step,
                    "content"
                ):

                    continue

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
            "Gemini đã xử lý nhưng không "
            "trả về nội dung văn bản."
        )

    except Exception as e:

        error_text = str(e)

        return (
            "❌ Không thể kết nối Gemini.\n\n"
            f"Chi tiết lỗi:\n{error_text}\n\n"
            "Hãy kiểm tra Gemini API Key "
            "và phiên bản google-genai."
        )


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================

if st.button(
    "🚀 Bắt đầu Quét & Phân tích",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # KIỂM TRA KEY
    # ========================================================

    if not api_key:

        st.error(
            "⚠️ Bạn chưa nhập Gemini API Key."
        )

        st.stop()

    api_key = api_key.strip()

    if not api_key:

        st.error(
            "⚠️ Gemini API Key không được để trống."
        )

        st.stop()

    # ========================================================
    # BƯỚC 1
    # ========================================================

    with st.spinner(
        "1/3: Đang tải dữ liệu WTI và tính RSI/MACD..."
    ):

        df, error_message = (
            get_technical_data()
        )

    if df is None:

        st.error(
            error_message
        )

        st.stop()

    # ========================================================
    # LẤY GIÁ TRỊ CUỐI
    # ========================================================

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

        values = [
            last_price,
            last_rsi,
            last_macd,
            last_signal
        ]

        if any(
            pd.isna(value)
            for value in values
        ):

            st.error(
                "❌ Dữ liệu kỹ thuật không hợp lệ."
            )

            st.stop()

    except Exception as e:

        st.error(
            f"❌ Không đọc được dữ liệu kỹ thuật: {e}"
        )

        st.stop()

    # ========================================================
    # BƯỚC 2
    # ========================================================

    with st.spinner(
        "2/3: Đang quét tin tức dầu mỏ toàn cầu..."
    ):

        news = get_global_news()

    # ========================================================
    # HIỂN THỊ THỊ TRƯỜNG
    # ========================================================

    st.subheader(
        "📊 Trạng thái thị trường"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Giá WTI",
        f"${last_price:.2f}"
    )

    col2.metric(
        "RSI",
        f"{last_rsi:.2f}"
    )

    # RSI status
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

    # MACD
    if last_macd > last_signal:

        macd_label = "🟢 Tăng"

    elif last_macd < last_signal:

        macd_label = "🔴 Giảm"

    else:

        macd_label = "🟡 Trung tính"

    col4.metric(
        "MACD",
        macd_label
    )

    # ========================================================
    # BIỂU ĐỒ
    # ========================================================

    st.subheader(
        "📈 Biểu đồ giá WTI"
    )

    chart_data = df[
        ["Close"]
    ].copy()

    st.line_chart(
        chart_data,
        height=350
    )

    # ========================================================
    # BƯỚC 3: GEMINI
    # ========================================================

    with st.spinner(
        "3/3: Gemini đang phân tích thị trường..."
    ):

        analysis = ai_analyze_market(
            news_list=news,
            last_price=last_price,
            rsi=last_rsi,
            macd=last_macd,
            signal=last_signal,
            api_key=api_key
        )

    # ========================================================
    # HIỂN THỊ KẾT QUẢ AI
    # ========================================================

    st.subheader(
        "🤖 Báo cáo phân tích của AI"
    )

    st.markdown(
        analysis
    )

    # ========================================================
    # TIN TỨC
    # ========================================================

    st.subheader(
        "📰 Tin tức đầu vào"
    )

    with st.expander(
        "Xem các tin tức đã thu thập"
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
    # THÔNG TIN
    # ========================================================

    st.markdown("---")

    st.caption(
        "Giá: Yahoo Finance / WTI Futures (CL=F) | "
        "Tin tức: DuckDuckGo News | "
        "AI: Google Gemini Interactions API"
    )
