import streamlit as st
import yfinance as yf
from duckduckgo_search import DDGS
import google.generativeai as genai
import pandas as pd
from datetime import datetime

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Dự báo Dầu mỏ Pro", layout="wide")
st.title("🛢️ Hệ thống AI Dự báo Giá Dầu (Macro + Technical)")
st.markdown("Tự động thu thập tin tức toàn cầu, phân tích chỉ báo kỹ thuật và suy luận cung/cầu.")

# Form nhập API Key ở sidebar
api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")
st.sidebar.markdown("[Lấy API Key miễn phí tại đây](https://aistudio.google.com/)")

def get_technical_data(ticker="CL=F", days=60):
    """Lấy dữ liệu giá dầu thô WTI và tính toán chỉ báo RSI, MACD"""
    data = yf.download(ticker, period=f"{days}d")
    if data.empty:
        return None
    
    # Tính toán chỉ báo RSI (14 ngày)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Tính toán MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data.tail(30)

def get_global_news(query="crude oil OR OPEC OR US inventory", max_results=15):
    """Dò quét tin tức thị trường mới nhất"""
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=max_results)
            return [res['title'] for res in results]
    except Exception as e:
        return ["Không thể lấy tin tức lúc này. Hệ thống có thể đang quá tải."]

def ai_analyze_market(news_list, last_price, rsi, macd, signal):
    """Kích hoạt Khối óc LLM để suy luận"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")
    
    news_text = "\n".join([f"- {n}" for n in news_list])
    prompt = f"""
    Bạn là một chuyên gia giao dịch hàng hóa và phân tích định lượng. Hãy phân tích các thông số sau và đưa ra dự báo hướng đi của giá dầu WTI:
    
    1. Giá hiện tại: ${last_price:.2f}
    2. Chỉ báo RSI: {rsi:.2f} (Dưới 30: Quá bán, Trên 70: Quá mua)
    3. Trạng thái MACD: MACD = {macd:.2f}, Đường Tín hiệu = {signal:.2f}
    4. Tin tức vĩ mô mới nhất:
    {news_text}
    
    Vui lòng xuất báo cáo theo đúng cấu trúc sau (không dài dòng):
    - 🎯 XU HƯỚNG DỰ KIẾN: (Tăng / Giảm / Đi ngang)
    - 📰 TÁC ĐỘNG VĨ MÔ: (Tổng hợp ngắn gọn nguồn cung/cầu từ tin tức)
    - 📊 TÍN HIỆU KỸ THUẬT: (Phe mua hay phe bán đang chiếm ưu thế trên biểu đồ)
    - 💡 CHIẾN LƯỢC GIAO DỊCH: (Điểm vào lệnh hoặc rủi ro cần chú ý)
    """
    response = model.generate_content(prompt)
    return response.text

# --- LUỒNG THỰC THI CHÍNH ---
if st.button("Bắt đầu Quét & Phân tích Tự động", type="primary"):
    if not api_key:
        st.error("⚠️ Bạn cần nhập Gemini API Key ở menu bên trái để hệ thống có thể tư duy.")
    else:
        with st.spinner("1/3: Đang tải dữ liệu biểu đồ và tính toán chỉ báo toán học..."):
            df = get_technical_data()
            last_price = float(df['Close'].iloc[-1].item() if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1])
            last_rsi = float(df['RSI'].iloc[-1].item() if isinstance(df['RSI'].iloc[-1], pd.Series) else df['RSI'].iloc[-1])
            last_macd = float(df['MACD'].iloc[-1].item() if isinstance(df['MACD'].iloc[-1], pd.Series) else df['MACD'].iloc[-1])
            last_signal = float(df['Signal'].iloc[-1].item() if isinstance(df['Signal'].iloc[-1], pd.Series) else df['Signal'].iloc[-1])
            
        with st.spinner("2/3: Đang dò quét tin tức địa chính trị và hàng hóa toàn cầu..."):
            news = get_global_news()
            
        st.subheader("Trạng thái Biểu đồ Kỹ thuật")
        col1, col2, col3 = st.columns(3)
        col1.metric("Giá WTI Hiện tại", f"${last_price:.2f}")
        col2.metric("Chỉ báo RSI", f"{last_rsi:.2f}")
        
        macd_status = "Tích cực (Cắt lên)" if last_macd > last_signal else "Tiêu cực (Cắt xuống)"
        col3.metric("Trạng thái MACD", macd_status)
        
        with st.spinner("3/3: AI đang kết hợp dữ liệu và suy luận logic..."):
            analysis = ai_analyze_market(news, last_price, last_rsi, last_macd, last_signal)
            
        st.subheader("Bản Báo Cáo Của AI")
        st.info(analysis)
        
        st.subheader("Dữ liệu Tin tức Đầu vào")
        with st.expander("Xem các bản tin gốc đã thu thập"):
            for i, n in enumerate(news):
                st.write(f"{i+1}. {n}")
