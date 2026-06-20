import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "final_model_chitay_finetuned.keras"
IMG_SIZE = 128

# ==========================
# CSS - MÀU CAM CHỦ ĐẠO
# ==========================
st.set_page_config(
    page_title="Palm Reading AI - Bói Vân Tay",
    page_icon="✋",
    layout="wide"
)

st.markdown("""
    <style>
    /* Màu cam chủ đạo */
    .main-header {
        background: linear-gradient(135deg, #FF6B00, #FF8C42);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(255, 107, 0, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
        letter-spacing: 2px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin: 10px 0 0 0;
        font-weight: 300;
    }
    
    .main-header .sub-info {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        margin-top: 8px;
        background: rgba(0,0,0,0.15);
        display: inline-block;
        padding: 5px 20px;
        border-radius: 20px;
    }
    
    /* Khung ảnh */
    .image-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 3px solid #FF8C42;
        box-shadow: 0 4px 20px rgba(255, 107, 0, 0.15);
        margin-bottom: 15px;
    }
    
    .image-container h4 {
        color: #E55A00;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    /* Nút bấm cam */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B00, #FF8C42);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 0, 0.3);
        width: 100%;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(255, 107, 0, 0.5);
        background: linear-gradient(135deg, #E55A00, #FF6B00);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Khung kết quả */
    .result-box {
        background: linear-gradient(135deg, #FFF3EB, #FFE4D6);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #FF6B00;
        box-shadow: 0 2px 15px rgba(255, 107, 0, 0.1);
        margin: 15px 0;
        min-height: 150px;
    }
    
    .result-box h3 {
        color: #E55A00;
        font-weight: 700;
        margin-bottom: 15px;
        font-size: 1.5rem;
    }
    
    /* Thẻ chỉ số */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #FF8C42;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-card .label {
        color: #666;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .metric-card .value {
        color: #FF6B00;
        font-size: 2rem;
        font-weight: 800;
        margin: 5px 0;
    }
    
    .metric-card .bar {
        background: #f0f0f0;
        border-radius: 10px;
        height: 8px;
        margin: 8px 0;
        overflow: hidden;
    }
    
    .metric-card .bar-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #FF8C42, #FF6B00);
        transition: width 1s ease;
    }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 4px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-high {
        background: #28a745;
        color: white;
    }
    
    .badge-medium {
        background: #ffc107;
        color: #333;
    }
    
    .badge-low {
        background: #dc3545;
        color: white;
    }
    
    /* Sidebar */
    .sidebar-info {
        background: #FFF3EB;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FF6B00;
        margin: 10px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.85rem;
        padding: 20px;
        margin-top: 30px;
        border-top: 2px solid #FFE4D6;
    }
    
    .footer strong {
        color: #FF6B00;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("""
    <div class="main-header">
        <h1>✋ PALM READING AI SYSTEM</h1>
        <p>Hệ thống bói vân tay thông minh với trí tuệ nhân tạo</p>
        <div class="sub-info">
            🧠 MobileNetV2 + Transfer Learning + Fine Tuning + Expert Rule Engine
        </div>
    </div>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    """Tải model AI đã được fine-tuning"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy file model: {MODEL_PATH}")
        st.info("💡 Vui lòng đảm bảo file model có trong thư mục hiện tại")
        return None
    
    with st.spinner("⏳ Đang tải model AI ..."):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            st.success("✅ Model đã được tải thành công!")
            return model
        except Exception as e:
            st.error(f"❌ Lỗi khi tải model: {str(e)}")
            return None

# ==========================
# PREPROCESS
# ==========================
def preprocess_image(img):
    """
    Tiền xử lý ảnh đầu vào:
    - Chuyển sang grayscale
    - CLAHE tăng cường độ tương phản
    - Resize về 128x128
    - Chuẩn hóa về [0,1]
    """
    # Chuyển RGB sang grayscale nếu cần
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # CLAHE - Tăng cường độ tương phản để làm rõ các đường chỉ tay
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Lưu ảnh đã xử lý để hiển thị
    display_img = gray.copy()
    
    # Resize về kích thước model yêu cầu
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # Chuẩn hóa về [0,1]
    gray = gray.astype(np.float32) / 255.0
    
    # Tạo 3 channels (vì model dùng RGB)
    gray = np.stack([gray, gray, gray], axis=-1)
    
    # Thêm batch dimension
    gray = np.expand_dims(gray, axis=0)
    
    return gray, display_img

# ==========================
# EXPERT SYSTEM - HỆ CHUYÊN GIA
# ==========================
def generate_report(sd, td, tam, sn):
    """
    Hệ thống suy luận dựa trên các quy tắc chuyên gia (Expert Rule Engine)
    Phân tích và đưa ra nhận định dựa trên 4 chỉ số:
    - sd: Sinh đạo (Life Line)
    - td: Trí đạo (Head Line)
    - tam: Tâm đạo (Heart Line)
    - sn: Sự nghiệp (Career Line)
    """
    report = []
    
    # Điểm tổng thể (weighted average)
    overall = (sd * 0.25 + td * 0.30 + tam * 0.20 + sn * 0.25)
    
    # ===== PHÂN TÍCH SỰ NGHIỆP =====
    if td > 0.7 and sn > 0.7:
        report.append("🔹 **Sự Nghiệp & Trí Tuệ:** Có xu hướng phù hợp với các lĩnh vực kỹ thuật, công nghệ hoặc quản lý. Tư duy logic tốt kết hợp với khả năng hoạch định chiến lược sẽ giúp bạn đạt được nhiều thành công trong sự nghiệp.")
    elif tam > 0.7 and td < 0.5:
        report.append("🔹 **Sự Nghiệp & Trí Tuệ:** Có thiên hướng nghệ thuật, sáng tạo và giao tiếp. Bạn phù hợp với các công việc liên quan đến truyền thông, nghệ thuật, hoặc các lĩnh vực đòi hỏi sự đồng cảm và kết nối với con người.")
    elif sn > 0.6:
        report.append("🔹 **Sự Nghiệp:** Đường sự nghiệp khá rõ nét, cho thấy bạn có định hướng và mục tiêu trong công việc. Hãy tiếp tục phát huy thế mạnh của mình để đạt được thành công.")
    else:
        report.append("🔹 **Sự Nghiệp:** Đường sự nghiệp chưa thực sự rõ ràng, có thể bạn đang trong giai đoạn tìm kiếm hướng đi cho bản thân. Đừng ngại thử nghiệm và khám phá các lĩnh vực mới.")
    
    # ===== PHÂN TÍCH TÍNH CÁCH =====
    if td > 0.7 and tam > 0.7:
        report.append("🔹 **Tính Cách & Cảm Xúc:** Khả năng cân bằng giữa lý trí và cảm xúc khá tốt. Bạn vừa có tư duy phân tích sắc bén, vừa có sự nhạy cảm và thấu hiểu cảm xúc của người khác. Đây là một sự kết hợp tuyệt vời.")
    elif td > tam:
        report.append("🔹 **Tính Cách & Cảm Xúc:** Có xu hướng suy nghĩ logic và phân tích. Bạn thường đưa ra quyết định dựa trên lý trí và dữ liệu cụ thể. Tuy nhiên, hãy nhớ lắng nghe cảm xúc của mình nhiều hơn.")
    elif tam > td:
        report.append("🔹 **Tính Cách & Cảm Xúc:** Có xu hướng đưa ra quyết định dựa trên cảm xúc và trực giác. Bạn là người giàu cảm xúc, sống tình cảm và có sự đồng cảm sâu sắc với mọi người xung quanh.")
    else:
        report.append("🔹 **Tính Cách & Cảm Xúc:** Sự cân bằng giữa lý trí và cảm xúc ở mức trung bình. Bạn có khả năng điều chỉnh linh hoạt giữa hai yếu tố này tùy theo hoàn cảnh.")
    
    # ===== PHÂN TÍCH SỨC KHỎE =====
    if sd > 0.8:
        report.append("🔹 **Sức Khỏe:** Sinh đạo cao cho thấy sức bền và khả năng thích nghi tốt. Bạn có nền tảng sức khỏe dồi dào, khả năng phục hồi nhanh chóng sau bệnh tật và có thể duy trì năng lượng lâu dài.")
    elif sd > 0.5:
        report.append("🔹 **Sức Khỏe:** Thể trạng ở mức khá ổn định. Bạn nên duy trì lối sống lành mạnh, tập thể dục đều đặn và chế độ ăn uống hợp lý để bảo vệ sức khỏe lâu dài.")
    else:
        report.append("🔹 **Sức Khỏe:** Cần quan tâm đặc biệt đến sức khỏe. Nên cân bằng giữa học tập, làm việc và nghỉ ngơi. Đừng quên dành thời gian cho các hoạt động thư giãn và tái tạo năng lượng.")
    
    # ===== GỢI Ý PHÁT TRIỂN =====
    if overall > 0.8:
        report.append("🌟 **Gợi ý phát triển:** Các chỉ số đều ở mức cao và khá cân bằng. Bạn đang trên đà phát triển tốt. Hãy tiếp tục nuôi dưỡng những điểm mạnh và đừng ngần ngại khám phá những tiềm năng mới. Tương lai đang rộng mở với bạn!")
    elif overall > 0.6:
        report.append("🌟 **Gợi ý phát triển:** Có nhiều tiềm năng phát triển trong tương lai. Hãy tập trung vào việc cải thiện các điểm yếu và phát huy tối đa thế mạnh của mình. Mỗi ngày đều là cơ hội để bạn tiến bộ.")
    elif overall > 0.4:
        report.append("🌟 **Gợi ý phát triển:** Kết quả cho thấy còn nhiều yếu tố cần cải thiện. Đừng nản lòng, hãy xem đây là những cơ hội để bạn phát triển bản thân. Xác định rõ mục tiêu và từng bước thực hiện chúng.")
    else:
        report.append("🌟 **Gợi ý phát triển:** Bạn đang trong giai đoạn xây dựng nền tảng. Hãy kiên nhẫn và tập trung vào việc phát triển các kỹ năng cơ bản. Thành công đến từ sự nỗ lực không ngừng và học hỏi mỗi ngày.")
    
    # ===== XÁC ĐỊNH MỨC ĐỘ =====
    if overall > 0.8:
        level = "🌟 Rất Tốt"
        badge_class = "badge-high"
    elif overall > 0.6:
        level = "👍 Khá"
        badge_class = "badge-medium"
    elif overall > 0.4:
        level = "📊 Trung Bình"
        badge_class = "badge-medium"
    else:
        level = "🔄 Cần Cải Thiện"
        badge_class = "badge-low"
    
    # ===== TẠO BÁO CÁO HOÀN CHỈNH =====
    final_text = f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #FF6B00;">
        <h3 style="color: #E55A00; margin-top: 0;">📊 ĐÁNH GIÁ TỔNG QUAN: <span class="badge {badge_class}">{level}</span></h3>
        <hr style="border: 1px solid #FFE4D6;">
        <div style="margin-top: 15px;">
            {"<br>".join(report)}
        </div>
        <hr style="border: 1px solid #FFE4D6;">
        <p style="color: #999; font-size: 0.85rem; margin-top: 10px;">
            ⚠️ <em>Kết quả mang tính tham khảo, dựa trên phân tích đường chỉ tay và hệ thống chuyên gia.</em>
        </p>
    </div>
    """
    
    return overall, final_text

# ==========================
# PREDICT FUNCTION
# ==========================
def predict_hand(img, model):
    """
    Hàm dự đoán chính:
    1. Tiền xử lý ảnh
    2. Dự đoán với model
    3. Tạo báo cáo với expert system
    """
    if img is None or model is None:
        return None, 0, 0, 0, 0, 0, "⚠️ Vui lòng chọn ảnh và đảm bảo model đã được tải."
    
    try:
        # Tiền xử lý ảnh
        x, processed_img = preprocess_image(img)
        
        # Dự đoán với model
        pred = model.predict(x, verbose=0)[0]
        
        # Lấy các chỉ số
        sd = float(pred[0])  # Sinh đạo
        td = float(pred[1])  # Trí đạo
        tam = float(pred[2]) # Tâm đạo
        sn = float(pred[3])  # Sự nghiệp
        
        # Tạo báo cáo
        overall, report_text = generate_report(sd, td, tam, sn)
        
        return (
            processed_img,
            round(sd * 100, 1),
            round(td * 100, 1),
            round(tam * 100, 1),
            round(sn * 100, 1),
            round(overall * 100, 1),
            report_text
        )
    
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
        return None, 0, 0, 0, 0, 0, f"❌ Có lỗi xảy ra: {str(e)}"

# ==========================
# MAIN UI
# ==========================

# Load model
model = load_model()

# Kiểm tra nếu model load thất bại
if model is None:
    st.warning("⚠️ Không thể tải model. Vui lòng kiểm tra file model.")
    st.stop()

# Layout 2 cột
col_left, col_right = st.columns([1, 1])

# ===== CỘT TRÁI: INPUT =====
with col_left:
    st.markdown("### 📸 Tải ảnh bàn tay")
    st.markdown("""
    <div style="background: #FFF3EB; padding: 10px; border-radius: 8px; margin-bottom: 15px; font-size: 0.9rem;">
        💡 Hướng dẫn: Tải ảnh bàn tay hoặc chụp trực tiếp từ webcam. 
        Đảm bảo ảnh rõ nét, có đủ ánh sáng để hệ thống phân tích chính xác.
    </div>
    """, unsafe_allow_html=True)
    
    # Image input
    uploaded_file = st.file_uploader(
        "Chọn ảnh từ máy tính (JPG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    # Hoặc camera
    camera_photo = st.camera_input(
        "Hoặc chụp trực tiếp từ camera",
        key="camera"
    )
    
    # Lấy ảnh từ file upload hoặc camera
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_np = np.array(img)
        st.session_state['input_image'] = img_np
        st.image(img, caption="✅ Ảnh đã tải lên", use_column_width=True)
    elif camera_photo is not None:
        img = Image.open(camera_photo)
        img_np = np.array(img)
        st.session_state['input_image'] = img_np
        st.image(img, caption="✅ Ảnh đã chụp", use_column_width=True)
    else:
        st.session_state['input_image'] = None
    
    # Nút dự đoán
    if st.button("🔍 PHÂN TÍCH VÂN TAY", use_container_width=True):
        if st.session_state.get('input_image') is not None:
            with st.spinner("⏳ AI đang phân tích... Hệ thống đang tiền xử lý ảnh và chạy model..."):
                result = predict_hand(st.session_state['input_image'], model)
                if result:
                    (processed_img, sd, td, tam, sn, overall, report) = result
                    st.session_state['result'] = {
                        'processed_img': processed_img,
                        'sd': sd, 'td': td, 'tam': tam, 'sn': sn,
                        'overall': overall, 'report': report
                    }
                    st.success("✅ Phân tích hoàn tất!")
        else:
            st.warning("⚠️ Vui lòng tải ảnh hoặc chụp ảnh trước khi phân tích!")

# ===== CỘT PHẢI: KẾT QUẢ =====
with col_right:
    st.markdown("### 🖐️ Kết quả phân tích")
    
    # Hiển thị ảnh đã xử lý
    if 'result' in st.session_state and st.session_state['result'] is not None:
        result_data = st.session_state['result']
        
        # Ảnh đã tiền xử lý
        if result_data['processed_img'] is not None:
            st.image(
                result_data['processed_img'],
                caption="📊 Ảnh sau tiền xử lý (CLAHE)",
                use_column_width=True,
                clamp=True
            )
        
        # Các chỉ số
        st.markdown("---")
        st.markdown("#### 📈 Chỉ số chi tiết")
        
        # Tạo 4 cột cho 4 đường
        col1, col2 = st.columns(2)
        
        with col1:
            # Sinh đạo
            sd = result_data['sd']
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">💪 Sinh Đạo</div>
                <div class="value">{sd}%</div>
                <div class="bar"><div class="bar-fill" style="width:{sd}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tâm đạo
            tam = result_data['tam']
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">❤️ Tâm Đạo</div>
                <div class="value">{tam}%</div>
                <div class="bar"><div class="bar-fill" style="width:{tam}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Trí đạo
            td = result_data['td']
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">🧠 Trí Đạo</div>
                <div class="value">{td}%</div>
                <div class="bar"><div class="bar-fill" style="width:{td}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sự nghiệp
            sn = result_data['sn']
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">💼 Sự Nghiệp</div>
                <div class="value">{sn}%</div>
                <div class="bar"><div class="bar-fill" style="width:{sn}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Điểm tổng thể
        st.markdown("---")
        overall_score = result_data['overall']
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FF6B00, #FF8C42); padding: 15px; border-radius: 12px; text-align: center; margin: 10px 0;">
            <span style="color: white; font-size: 1.2rem; font-weight: 600;">📊 ĐIỂM TỔNG THỂ</span>
            <div style="color: white; font-size: 3rem; font-weight: 800;">{overall_score}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Báo cáo chi tiết
        st.markdown("---")
        st.markdown("#### 📝 Báo cáo phân tích")
        st.markdown(result_data['report'], unsafe_allow_html=True)
        
        # Nút reset
        if st.button("🔄 Làm mới kết quả", use_container_width=True):
            st.session_state['result'] = None
            st.session_state['input_image'] = None
            st.rerun()
    
    else:
        # Placeholder khi chưa có kết quả
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px; color: #999;">
            <p style="font-size: 4rem;">🔮</p>
            <p style="font-size: 1.2rem; font-weight: 500;">Chưa có kết quả</p>
            <p style="font-size: 0.9rem;">Tải ảnh bàn tay và nhấn <strong>"PHÂN TÍCH VÂN TAY"</strong></p>
            <p style="font-size: 0.85rem; color: #ccc;">Hệ thống sẽ phân tích 4 đường chỉ tay chính</p>
        </div>
        """, unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🤖 Hệ thống AI")
    st.markdown("""
    <div class="sidebar-info">
        <strong>Model:</strong> MobileNetV2<br>
        <strong>Phương pháp:</strong> Transfer Learning + Fine Tuning<br>
        <strong>Expert System:</strong> Rule-based reasoning
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📖 Hướng dẫn")
    st.markdown("""
    **1.** Tải ảnh bàn tay (upload hoặc chụp webcam)
    
    **2.** Nhấn **"PHÂN TÍCH VÂN TAY"**
    
    **3.** Xem kết quả:
    - 🖐️ Ảnh đã tiền xử lý
    - 📈 4 chỉ số đường chỉ tay
    - 📝 Báo cáo phân tích chi tiết
    """)
    
    st.markdown("---")
    st.markdown("### 🖐️ 4 đường chỉ tay")
    st.markdown("""
    | Đường | Ý nghĩa |
    |-------|---------|
    | 💪 Sinh đạo | Sức khỏe, sinh lực |
    | 🧠 Trí đạo | Trí tuệ, tư duy |
    | ❤️ Tâm đạo | Tình cảm, cảm xúc |
    | 💼 Sự nghiệp | Công việc, thành công |
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Lưu ý")
    st.markdown("""
    - Kết quả mang tính **tham khảo**
    - Đảm bảo ảnh **rõ nét**, đủ sáng
    - Ảnh càng rõ, độ chính xác càng cao
    """)
    
    st.markdown("---")
    st.markdown(f"""
    <div style="background: #FFF3EB; padding: 15px; border-radius: 10px; text-align: center;">
        <p style="margin: 0; color: #E55A00; font-weight: 600;">🔶 Màu cam chủ đạo</p>
        <p style="font-size: 0.85rem; color: #666; margin: 5px 0 0 0;">
            Version 1.0 | Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
<div class="footer">
    Made with ❤️ | <strong>Palm Reading AI</strong> | 
    MobileNetV2 + Transfer Learning + Fine Tuning + Expert System
    <br>
    <span style="font-size: 0.8rem;">© 2024 - Kết quả mang tính tham khảo</span>
</div>
""", unsafe_allow_html=True)
