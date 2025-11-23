import cv2
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# === Configuração da Página ===
st.set_page_config(page_title="Tony Hand-Skeletor", page_icon="🖐️", layout="centered")

st.title("🖐️ Detector de Mãos - Tony Skeletor")
st.caption("Rastreamento de 21 pontos biomecânicos em tempo real.")

# === Sidebar ===
with st.sidebar:
    st.header("Configurações Neurais")
    confianca = st.slider("Sensibilidade de Detecção", 0.1, 1.0, 0.5)
    st.info("Quanto maior a sensibilidade, mais certeza a IA precisa ter para desenhar a mão. Se estiver falhando, diminua.")
    st.markdown("---")
    st.markdown("### 👑 Tecnologia Google MediaPipe + Tony")

# === Inicialização do MediaPipe ===
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Inicializa o modelo de mãos FORA do loop para não travar a memória
hands_processor = mp_hands.Hands(
    model_complexity=0,  # 0 é mais rápido (bom para celular), 1 é mais preciso
    min_detection_confidence=confianca,
    min_tracking_confidence=confianca,
    max_num_hands=2
)

# === Processador de Vídeo ===
class HandDetector:
    def recv(self, frame):
        # 1. Converte o frame do WebRTC (av) para OpenCV (numpy)
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Converte BGR para RGB (O MediaPipe só enxerga RGB)
        img.flags.writeable = False # Otimização de performance
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. A Mágica: Detecta as mãos
        results = hands_processor.process(img_rgb)
        
        # 4. Desenha o esqueleto se achar mãos
        img.flags.writeable = True
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os pontos (juntas) e conexões (ossos)
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
        # 5. Retorna o frame desenhado
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# === Interface ===
st.markdown("### 🧬 Ativar Visão Biomecânica")
st.info("Levante as mãos para a câmera. Funciona melhor com boa iluminação.")

# Streamer WebRTC
webrtc_streamer(
    key="hand-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=HandDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.markdown("**Powered by Python & MediaPipe**")
