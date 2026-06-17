import streamlit as st
import cv2
import numpy as np

# -----------------------------
# 1. CONFIGURAÇÃO DO APP
# -----------------------------
st.set_page_config(
    page_title="NucleoClass OS",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ NucleoClass OS")
st.subheader("Índice Densitométrico Contínuo de Catarata Nuclear")
st.markdown("---")

# -----------------------------
# 2. UPLOAD DA IMAGEM
# -----------------------------
arquivo_upload = st.file_uploader(
    "Envie uma imagem da lâmpada de fenda",
    type=["jpg", "jpeg", "png"]
)

if arquivo_upload is not None:

    # Leitura da imagem em Tons de Cinza (0 a 255)
    file_bytes = np.asarray(bytearray(arquivo_upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Imagem carregada", use_column_width=True)

    with st.spinner("Processando imagem com filtros densitométricos..."):

        # -----------------------------
        # 3. ROI (REGIÃO DE INTERESSE) - Focada no feixe central
        # -----------------------------
        h, w = img.shape
        roi = img[int(h*0.3):int(h*0.7), int(w*0.4):int(w*0.65)]

        # -----------------------------
        # 4. PRÉ-PROCESSAMENTO
        # -----------------------------
        roi_suave = cv2.GaussianBlur(roi, (7, 7), 0)

        # -----------------------------
        # 5. REMOÇÃO DE REFLEXOS (Filtro de Exclusão de Córnea/Flash)
        # -----------------------------
        # Cortamos pixels acima de 240 para o reflexo Purkinje não mascarar o cristalino
        mascara = roi_suave[roi_suave < 240]

        if len(mascara) == 0:
            mascara = roi_suave.flatten()

        # -----------------------------
        # 6. EXTRAÇÃO DE FEATURES REAIS
        # -----------------------------
        media = np.mean(mascara)
        mediana = np.median(mascara)
        desvio = np.std(mascara)
        p25 = np.percentile(mascara, 25)
        p75 = np.percentile(mascara, 75)
        contraste = p75 - p25

        # -----------------------------
        # 7. ÍNDICE NC (ESCALA FIXA PADRONIZADA)
        # -----------------------------
        # Calibração baseada na escala absoluta de cinza da câmera (0-255)
        # Onde 25 é o corte mínimo de ruído de fundo e 190 é o teto máximo de esclerose nuclear
        min_absoluto = 25.0
        max_absoluto = 190.0

        # Cálculo da rampa contínua real
        nc_index = ((mediana - min_absoluto) / (max_absoluto - min_absoluto)) * 100
        nc_index = float(np.clip(nc_index, 0, 100))
        nc_index = round(nc_index, 2)

        # -----------------------------
        # 8. RESULTADO
        # -----------------------------
        st.markdown("## 📊 NC-Index")
        st.metric("Índice Densitométrico", f"{nc_index} / 100")

        # -----------------------------
        # 9. FEATURES (Essencial para a análise estatística do seu TCC)
        # -----------------------------
        with st.expander("Ver features extraídas para análise de dados"):
            st.write({
                "Média de Brilho Absoluto": float(media),
                "Mediana de Brilho Absoluto": float(mediana),
                "Desvio padrão": float(desvio),
                "Percentil 25 (P25)": float(p25),
                "Percentil 75 (P75)": float(p75),
                "Contraste da fenda": float(contraste)
            })

        # -----------------------------
        # 10. CATEGORIZAÇÃO CLÍNICA DERIVADA
        # -----------------------------
        st.markdown("## 🧠 Interpretação clínica")

        if nc_index < 35.0:
            st.success(f"🟢 NC leve ({nc_index}) — compatível com acompanhamento clínico")
        elif nc_index < 65.0:
            st.warning(f"🟡 NC moderado ({nc_index}) — monitoramento periódico")
        else:
            st.error(f"🔴 NC avançado ({nc_index}) — sugerir programação cirúrgica prioritária")

        # -----------------------------
        # 11. INTERPRETAÇÃO METODOLÓGICA
        # -----------------------------
        st.markdown("---")
        st.caption(
            "Análise baseada em refletividade absoluta de pixels (Filtro Anti-Reflexo Purkinje ativo). "
            "Exige estrita padronização da lâmpada de fenda (Canon Rebel T7, 16x de magnificação, fenda a 45°)."
        )
