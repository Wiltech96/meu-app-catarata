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

    # leitura da imagem
    file_bytes = np.asarray(bytearray(arquivo_upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Imagem carregada", use_column_width=True)

    with st.spinner("Processando imagem..."):

        # -----------------------------
        # 3. ROI (REGIÃO DE INTERESSE)
        # -----------------------------
        h, w = img.shape

        roi = img[int(h*0.3):int(h*0.7), int(w*0.35):int(w*0.65)]

        # -----------------------------
        # 4. PRÉ-PROCESSAMENTO
        # -----------------------------
        roi_suave = cv2.GaussianBlur(roi, (5, 5), 0)

        # -----------------------------
        # 5. REMOÇÃO DE REFLEXOS (robusta)
        # -----------------------------
        limite_reflexo = np.percentile(roi_suave, 99)
        mascara = roi_suave[roi_suave < limite_reflexo]

        if len(mascara) == 0:
            mascara = roi_suave.flatten()

        # -----------------------------
        # 6. EXTRAÇÃO DE FEATURES
        # -----------------------------
        media = np.mean(mascara)
        mediana = np.median(mascara)
        desvio = np.std(mascara)

        p25 = np.percentile(mascara, 25)
        p75 = np.percentile(mascara, 75)

        # contraste simples
        contraste = p75 - p25

        # -----------------------------
        # 7. ÍNDICE NC (NORMALIZADO)
        # -----------------------------
        # normalização baseada no próprio range da imagem (mais robusto)
        min_val = np.percentile(mascara, 5)
        max_val = np.percentile(mascara, 95)

        if max_val - min_val == 0:
            nc_index = 0
        else:
            nc_index = ((mediana - min_val) / (max_val - min_val)) * 100

        nc_index = float(np.clip(nc_index, 0, 100))
        nc_index = round(nc_index, 2)

        # -----------------------------
        # 8. RESULTADO
        # -----------------------------
        st.markdown("## 📊 NC-Index")

        st.metric("Índice Densitométrico", f"{nc_index} / 100")

        # -----------------------------
        # 9. FEATURES (para pesquisa)
        # -----------------------------
        with st.expander("Ver features extraídas"):
            st.write({
                "Média": float(media),
                "Mediana": float(mediana),
                "Desvio padrão": float(desvio),
                "P25": float(p25),
                "P75": float(p75),
                "Contraste": float(contraste)
            })

        # -----------------------------
        # 10. CATEGORIZAÇÃO CLÍNICA (derivada)
        # -----------------------------
        st.markdown("## 🧠 Interpretação clínica")

        if nc_index < 30:
            st.success(f"🟢 NC leve ({nc_index}) — acompanhamento clínico")
        elif nc_index < 60:
            st.warning(f"🟡 NC moderado ({nc_index}) — monitoramento periódico")
        else:
            st.error(f"🔴 NC avançado ({nc_index}) — considerar cirurgia")

        # -----------------------------
        # 11. INTERPRETAÇÃO METODOLÓGICA
        # -----------------------------
        st.markdown("---")
        st.caption(
            "Índice contínuo baseado em densitometria relativa da região nuclear. "
            "Valores dependem de padronização de captura (lâmpada de fenda, ISO, exposição).
