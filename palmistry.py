import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf  
import cv2

# ====== LOAD AI MODEL ======
# Thay đổi đường dẫn theo model của bạn
@st.cache_resource
def load_model():
    """
    Load model AI đã train
    """
    model = tf.keras.models.load_model('final_model_chitay_finetuned.h5')
    
    return model

# ====== HÀM TIỀN XỬ LÝ ẢNH ======
def preprocess_image(image):
    """
    Tiền xử lý ảnh trước khi đưa vào model
    """
    # Chuyển PIL sang numpy array
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    # Resize về kích thước model yêu cầu 
    img = cv2.resize(img, (200, 200))
    
    # Chuẩn hóa (tùy theo cách bạn train)
    img = img / 255.0  # Normalize về [0,1]
    
    # Thêm batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# ====== HÀM DỰ ĐOÁN ======
def predict_palm(image, model):
    """
    Dự đoán từ ảnh bàn tay
    """
    # Tiền xử lý ảnh
    processed_img = preprocess_image(image)
    
    # Dự đoán
    predictions = model.predict(processed_img)
    
    # ====== XỬ LÝ KẾT QUẢ ======
    # Giả sử model output là 4 classes (4 đường)
    # Và mỗi class có xác suất và mô tả
    
    # Ví dụ: model trả về xác suất cho 4 đường
    # [sinh_dao, tam_dao, tri_dao, su_nghiep]
    
    # Tạo kết quả mẫu (thay bằng logic xử lý output của model bạn)
    lines_data = {
        'Sinh đạo': {
            'score': float(predictions[0][0]),
            'description': 'Sức khỏe tốt, tuổi thọ cao' if predictions[0][0] > 0.5 else 'Cần chú ý sức khỏe'
        },
        'Tâm đạo': {
            'score': float(predictions[0][1]),
            'description': 'Tình cảm sâu sắc, chân thành' if predictions[0][1] > 0.5 else 'Khó bày tỏ cảm xúc'
        },
        'Trí đạo': {
            'score': float(predictions[0][2]),
            'description': 'Tư duy logic, học hỏi tốt' if predictions[0][2] > 0.5 else 'Thiên về trực giác'
        },
        'Sự nghiệp': {
            'score': float(predictions[0][3]),
            'description': 'Cơ hội thăng tiến tốt' if predictions[0][3] > 0.5 else 'Cần nỗ lực nhiều hơn'
        }
    }
    
    # Tổng kết dựa trên điểm số
    scores = [d['score'] for d in lines_data.values()]
    avg_score = sum(scores) / len(scores)
    
    if avg_score > 0.7:
        overall = '🌟 Tuyệt vời! Vận mệnh của bạn rất tốt đẹp!'
    elif avg_score > 0.5:
        overall = '✨ Tốt! Bạn có tiềm năng phát triển mạnh mẽ!'
    else:
        overall = '🌱 Còn nhiều cơ hội phía trước, hãy kiên trì!'
    
    return {
        'overall': overall,
        'lines': lines_data
    }

# ====== CẤU HÌNH TRANG ======
st.set_page_config(
    page_title="Palmistry - Bói Vân Tay AI",
    page_icon="✋",
    layout="wide"
)

# ====== CSS ======
st.markdown("""
    <style>
    .main-title {
        background: linear-gradient(135deg, #FF6B00, #FF8C42);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(255, 107, 0, 0.3);
    }
    
    .main-title h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-title p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 10px 0 0 0;
    }
    
    .camera-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #FF8C42;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FF6B00, #FF8C42);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(255, 107, 0, 0.4);
    }
    
    .result-box {
        background: linear-gradient(135deg, #FFF3EB, #FFE4D6);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #FF6B00;
        min-height: 200px;
        box-shadow: 0 2px 10px rgba(255, 107, 0, 0.1);
    }
    
    .score-bar {
        background: #f0f0f0;
        border-radius: 10px;
        height: 8px;
        margin: 5px 0;
        overflow: hidden;
    }
    
    .score-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #FF8C42, #FF6B00);
        transition: width 0.5s ease;
    }
    
    .line-card {
        background: white;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #FF6B00;
    }
    </style>
""", unsafe_allow_html=True)

# ====== HEADER ======
st.markdown("""
    <div class="main-title">
        <h1>✋ Bói Vân Tay - Palmistry AI</h1>
        <p>AI phân tích 4 đường chỉ tay: Sinh đạo, Tâm đạo, Trí đạo, Sự nghiệp</p>
    </div>
""", unsafe_allow_html=True)

# ====== SESSION STATE ======
if 'image_captured' not in st.session_state:
    st.session_state.image_captured = None
if 'result_text' not in st.session_state:
    st.session_state.result_text = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None

# ====== LOAD MODEL ======
with st.sidebar:
    st.markdown("### 🤖 Trạng thái AI")
    
    if st.button("🔄 Load AI Model", use_container_width=True):
        with st.spinner("⏳ Đang load model..."):
            try:
                st.session_state.model = load_model()
                st.session_state.model_loaded = True
                st.success("✅ Model loaded thành công!")
            except Exception as e:
                st.error(f"❌ Lỗi load model: {str(e)}")
    
    if st.session_state.model_loaded:
        st.success("✅ AI Model đã sẵn sàng")
    else:
        st.warning("⚠️ Chưa load model")
        st.info("💡 Nhấn nút trên để load model")

# ====== LAYOUT CHÍNH ======
col_left, col_right = st.columns([1, 1])

# ====== CỘT TRÁI: CHỤP ẢNH ======
with col_left:
    st.markdown("### 📸 Khu vực chụp ảnh")
    
    with st.container():
        st.markdown('<div class="camera-box">', unsafe_allow_html=True)
        
        option = st.radio(
            "Chọn cách lấy ảnh:",
            ["📷 Chụp từ Camera", "📁 Tải ảnh lên"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if option == "📷 Chụp từ Camera":
            camera_photo = st.camera_input(
                "Hướng camera vào lòng bàn tay",
                key="camera_input"
            )
            
            if camera_photo is not None:
                img = Image.open(camera_photo)
                st.session_state.image_captured = img
                st.image(img, caption="✅ Ảnh đã chụp", use_column_width=True)
        
        else:
            uploaded_file = st.file_uploader(
                "Chọn ảnh bàn tay (JPG, PNG)",
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                st.session_state.image_captured = img
                st.image(img, caption="✅ Ảnh đã tải lên", use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ====== NÚT CHỨC NĂNG ======
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn1:
        if st.button("🔮 Dự đoán", use_container_width=True):
            # Kiểm tra đã có ảnh chưa
            if st.session_state.image_captured is None:
                st.warning("⚠️ Vui lòng chụp hoặc tải ảnh trước")
            
            # Kiểm tra đã load model chưa
            elif not st.session_state.model_loaded:
                st.warning("⚠️ Vui lòng load AI model trước")
            
            else:
                with st.spinner("⏳ AI đang phân tích vân tay..."):
                    try:
                        # Gọi hàm predict với model
                        result = predict_palm(
                            st.session_state.image_captured,
                            st.session_state.model
                        )
                        st.session_state.result_text = result
                        st.success("✅ Dự đoán hoàn tất!")
                    except Exception as e:
                        st.error(f"❌ Lỗi dự đoán: {str(e)}")
    
    with col_btn2:
        if st.button("🔄 Làm mới", use_container_width=True):
            st.session_state.image_captured = None
            st.session_state.result_text = None
            st.rerun()
    
    with col_btn3:
        if st.button("🗑️ Xóa ảnh", use_container_width=True):
            st.session_state.image_captured = None
            st.rerun()

# ====== CỘT PHẢI: KẾT QUẢ ======
with col_right:
    st.markdown("### 📊 Kết quả dự đoán")
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    if st.session_state.result_text is not None:
        result = st.session_state.result_text
        
        # Tổng quan
        st.markdown("#### 🔮 Tổng quan")
        st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <p style="font-size: 1.1rem; color: #333;">{result['overall']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Chi tiết từng đường
        st.markdown("#### 🖐️ Chi tiết 4 đường")
        
        line_emojis = {
            'Sinh đạo': '💪',
            'Tâm đạo': '❤️',
            'Trí đạo': '🧠',
            'Sự nghiệp': '💼'
        }
        
        # Hiển thị dạng thanh score
        for line_name, data in result['lines'].items():
            score = data['score']
            description = data['description']
            
            # Xác định màu dựa trên score
            if score > 0.7:
                status = "🟢 Tốt"
            elif score > 0.4:
                status = "🟡 Trung bình"
            else:
                status = "🔴 Cần cải thiện"
            
            st.markdown(f"""
                <div class="line-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #E55A00;">{line_emojis.get(line_name, '')} {line_name}</strong>
                        <span style="font-size: 0.9rem; color: #666;">{status}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {score*100}%;"></div>
                    </div>
                    <p style="font-size: 0.9rem; margin-top: 8px; color: #555;">{description}</p>
                    <p style="font-size: 0.8rem; color: #999; margin: 0;">Độ tin cậy: {score*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Nút xuất kết quả
        st.markdown("---")
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📋 Sao chép kết quả", use_container_width=True):
                st.info("📋 Đã sao chép kết quả")
        
        with col_export2:
            if st.button("📥 Tải kết quả", use_container_width=True):
                # Tạo text để tải
                text = f"KẾT QUẢ BÓI VÂN TAY\n{'='*30}\n\n"
                text += f"{result['overall']}\n\n"
                for line_name, data in result['lines'].items():
                    text += f"{line_name}: {data['description']} (Độ tin cậy: {data['score']*100:.1f}%)\n"
                
                st.download_button(
                    label="📥 Tải file",
                    data=text,
                    file_name="ket_qua_boi_van_tay.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    else:
        # Placeholder
        st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #999;">
                <p style="font-size: 3rem;">🔮</p>
                <p style="font-size: 1.1rem;">Chưa có kết quả</p>
                <p style="font-size: 0.9rem;">Hãy chụp ảnh bàn tay và nhấn "Dự đoán"</p>
                <p style="font-size: 0.85rem; color: #ccc;">(Đảm bảo đã load AI model)</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ====== FOOTER ======
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #aaa; font-size: 0.85rem; padding: 10px;">
        Made with ❤️ | Palmistry AI v2.0 | Kết quả mang tính tham khảo
    </div>
""", unsafe_allow_html=True)