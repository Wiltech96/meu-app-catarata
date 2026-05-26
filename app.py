import streamlit as st
import cv2
import numpy as np

# 1. Configuração da Identidade Visual Médica do Software
st.set_page_config(
    page_title="NucleoClass Auto", 
    layout="centered", 
    page_icon="👁️"
)
st.title("👁️ NucleoClass - Automação por Varredura de Intensidade")
st.subheader("Módulo de Extração de Métricas Físicas e Calibração Dinâmica")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda Fina Vertical):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. VARREDURA INTELIGENTE: Encontra o centro óptico da fenda de luz automaticamente
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_busca = img_gray[int(altura*0.4):int(altura*0.6), :] # Busca na faixa central horizontal
    
    # Calcula a média de brilho vertical para achar a coluna (X) mais iluminada (o pico da fenda)
    perfil_luminoso = np.mean(roi_busca, axis=0)
    coluna_pico_x = int(np.argmax(perfil_luminoso))
    
    # Margem de segurança contra erros de borda
    if coluna_pico_x < int(largura*0.2) or coluna_pico_x > int(largura*0.8):
        coluna_pico_x = int(largura * 0.5)
        
    # 5. DEFINE A ROI AUTOMÁTICA CENTRADA NO PICO DA FENDA (Filete estreito vertical)
    ymin, ymax = int(altura * 0.40), int(altura * 0.60)
    xmin, xmax = max(0, coluna_pico_x - int(largura * 0.04)), min(largura, coluna_pico_x + int(largura * 0.04))
    
    # 6. PROCESSAMENTO DIGITAL DE SINAIS (Espaço HSV e RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_hsv = img_hsv[ymin:ymax, xmin:xmax]
    
    media_h = float(np.mean(roi_hsv[:, :, 0])) # Matiz
    media_s = float(np.mean(roi_hsv[:, :, 1])) # Saturação
    media_v = float(np.mean(roi_hsv[:, :, 2])) # Luminosidade/Brilho V
    
    # Extrai canais RGB originais para Razão Cromática
    canal_red = img[ymin:ymax, xmin:xmax, 2]
    canal_blue = img[ymin:ymax, xmin:xmax, 0]
    media_r = float(np.mean(canal_red))
    media_b = float(np.mean(canal_blue))
    razao_vermelho_azul = media_r / (media_b + 0.001)
    
    # 7. EXIBIÇÃO VISUAL DO ENQUADRAMENTO DA IA
    img_viz = img.copy()
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Retângulo Centralizado Automaticamente no Pico de Luz da Fenda", use_container_width=True)
    
    # 8. EXIBIÇÃO DAS MÉTRICAS REAIS DA IMAGEM
    st.markdown("### 📊 Dados Densitométricos Puros")
    st.write("Estes são os números reais que o sensor está extraindo de dentro da fenda iluminada:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Brilho Puro (Canal V)", f"{media_v:.1f}")
    with col2:
        st.metric("Saturação de Cor (Canal S)", f"{media_s:.1f}")
    with col3:
        st.metric("Razão Cromática (R/A)", f"{razao_vermelho_azul:.2f}")
        
    st.markdown("---")
    st.info("💡 **Como fechar o projeto:** Suba suas fotos reais (Branca, Rubra, G2, G3, G4) neste aplicativo, tire um print de cada tela mostrando os números gerados e me mande por aqui. Com esses valores reais, nós travamos os limites definitivos do classificador automático.")
