import streamlit as st
import cv2
import numpy as np

# 1. Configuração da página do aplicativo
st.set_page_config(page_title="NucleoClass OS", page_icon="👁️", layout="centered")

st.title("👁️ NucleoClass OS")
st.subheader("Análise Densitométrica Contínua de Catarata Nuclear")
st.markdown("---")

# 2. Botão de Upload original para testar as fotos
arquivo_upload = st.file_uploader("Arraste ou selecione uma foto da fenda óptica", type=["jpg", "jpeg", "png"])

if arquivo_upload is not None:
    # Ler a foto enviada e converter para tons de cinza
    file_bytes = np.asarray(bytearray(arquivo_upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Mostrar a foto na tela do seu app
    st.image(img, caption="Exame Carregado", use_column_width=True)
    
    with st.spinner("Processando matriz de pixels com filtros ativos..."):
        # --- MOTOR DO NUCLEOCLASS ATUALIZADO (BLINDADO CONTRA REFLEXOS) ---
        h, w = img.shape
        
        # Ajuste de Região de Interesse (ROI) focada na fenda central
        roi = img[int(h*0.3):int(h*0.7), int(w*0.4):int(w*0.65)]
        
        # Filtro Gaussiano para suavizar ruídos e artefatos de penumbra
        roi_suave = cv2.GaussianBlur(roi, (7, 7), 0)
        
        # FILTRO ANTI-REFLEXO: Remove os pontos ultra-brancos (brilho > 240) da córnea/flashes
        mascara_sem_reflexo = roi_suave[roi_suave < 240]
        
        # Segurança caso a imagem seja muito fora do padrão
        if len(mascara_sem_reflexo) > 0:
            brilho_real_cristalino = np.mean(mascara_sem_reflexo)
        else:
            brilho_real_cristalino = np.mean(roi_suave)
        
        # Calibração Fina adaptada para as curvas de ganho óptico da Canon Rebel T7
        min_calibracao, max_calibracao = 25.0, 190.0
        nc_index = max(0.0, min(100.0, ((brilho_real_cristalino - min_calibracao) / (max_calibracao - min_calibracao)) * 100))
        nc_index = round(nc_index, 2)
        
    # --- MOSTRAR RESULTADO ---
    st.markdown("### 📊 Resultado da Análise Digital")
    st.metric(label="Índice Densitométrico (NC-Index)", value=f"{nc_index} / 100")
    
    # Categorização clínica baseada na rampa contínua
    if nc_index < 35.0:
        st.success(f"🟢 **Densidade Leve ({nc_index}):** Compatível com acompanhamento clínico de rotina.")
    elif nc_index < 65.0:
        st.warning(f"🟡 **Densidade Moderada ({nc_index}):** Compatível com monitoramento periódico ambulatorial.")
    else:
        st.error(f"🔴 **Densidade Avançada ({nc_index}):** Alto risco estrutural. Sugere-se programação cirúrgica prioritária.")
