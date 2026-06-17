import streamlit as st
import cv2
import numpy as np

# Configuração da página do aplicativo
st.set_page_config(page_title="NucleoClass OS", page_icon="👁️", layout="centered")

st.title("👁️ NucleoClass OS")
st.subheader("Análise Densitométrica Contínua de Catarata Nuclear")
st.markdown("---")

# Botão de Upload para testar as fotos
arquivo_upload = st.file_uploader("Arraste ou selecione uma foto da fenda óptica", type=["jpg", "jpeg", "png"])

if arquivo_upload is not None:
    # Ler a foto enviada
    file_bytes = np.asarray(bytearray(arquivo_upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Mostrar a foto na tela do seu app
    st.image(img, caption="Exame Carregado", use_column_width=True)
    
    # --- MOTOR DO NUCLEOCLASS (Processamento de Pixels) ---
    h, w = img.shape
    # Isola a região central (Núcleo do cristalino)
    roi = img[int(h*0.2):int(h*0.8), int(w*0.35):int(w*0.65)]
    roi_suave = cv2.GaussianBlur(roi, (5, 5), 0)
    brilho_medio = np.mean(roi_suave)
    
    # Transforma o brilho na escala contínua de 0 a 100 (Normalização Relativa)
    min_calibracao, max_calibracao = 15.0, 220.0
    nc_index = max(0.0, min(100.0, ((brilho_medio - min_calibracao) / (max_calibracao - min_calibracao)) * 100))
    nc_index = round(nc_index, 2)
    
    # --- MOSTRAR RESULTADO ---
    st.markdown("### 📊 Resultado da Análise")
    st.metric(label="Índice Densitométrico (NC-Index)", value=f"{nc_index} / 100")
    
    if nc_index < 35.0:
        st.success(f"🟢 **Densidade Leve ({nc_index}):** Compatível com acompanhamento clínico.")
    elif nc_index < 65.0:
        st.warning(f"🟡 **Densidade Moderada ({nc_index}):** Compatível com monitoramento periódico.")
    else:
        st.error(f"🔴 **Densidade Avançada ({nc_index}):** Sugere programação cirúrgica prioritária.")
