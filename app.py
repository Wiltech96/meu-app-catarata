import streamlit as st
import cv2
import numpy as np

# 1. Configuração visual do aplicativo
st.set_page_config(page_title="CataractApp G0-G6", layout="centered", page_icon="https://flaticon.com")
st.title("👁️ Classificador de Catarata Ambulatorial (G0-G6)")
st.subheader("Sistema de Diagnóstico Instantâneo por Smartphone")
st.markdown("---")

# 2. Área de upload da imagem do paciente
arquivo = st.file_uploader("Insira a foto batida no smartphone:", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Lê a imagem (independente do tamanho do celular)
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. RECORTE PERCENTUAL DINÂMICO (Adaptado para as suas fotos de teste)
    ymin, ymax = int(altura * 0.40), int(altura * 0.60)
    xmin, xmax = int(largura * 0.40), int(largura * 0.70)
    
    # 5. Extração dos canais Vermelho e Azul
    canal_red = img[:, :, 2]
    canal_blue = img[:, :, 0]
    
    media_vermelho = np.mean(canal_red[ymin:ymax, xmin:xmax])
    media_azul = np.mean(canal_blue[ymin:ymax, xmin:xmax])
    
    # 6. Desenha o retângulo verde para conferência do médico
    img_viz = img.copy()
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Área de Leitura do Núcleo", use_container_width=True)
    
    # 7. MOTOR DE DECISÃO INTELIGENTE (Régua do TCC Corrigida)
    if media_azul >= 130.0 and media_vermelho > 100.0:
        laudo, cor, conduta = "G5 - Catarata Branca / Total Intumescente", "red", "Opacificação total. Usar Azul de Tripano para capsulorréxis."
    elif media_azul < (media_vermelho / 3) and media_vermelho > 50.0:
        laudo, cor, conduta = "G6 - Catarata Rubra / Brunescente Ultra-Densa", "purple", "Alta densidade nuclear (rocha). Exige proteção endotelial máxima."
    else:
        if media_vermelho <= 45.0:
            laudo, cor, conduta = "G0 - Cristalino Transparente / Incipiente", "green", "Parâmetros mínimos de energia."
        elif media_vermelho <= 90.0:
            laudo, cor, conduta = "G1 - Grau I (Catarata Nuclear Inicial)", "green", "Fragmentação fácil. Energia padrão baixa."
        elif media_vermelho <= 135.0:
            laudo, cor, conduta = "G2 - Grau II (Catarata Nuclear Moderada-Leve)", "blue", "Procedimento cirúrgico padrão estável."
        elif media_vermelho <= 180.0:
            laudo, cor, conduta = "G3 - Grau III (Catarata Nuclear Moderada-Avançada)", "orange", "Núcleo denso. Considerar técnicas mecânicas (Faco-Chop)."
        else:
            laudo, cor, conduta = "G4 - Grau IV (Catarata Nuclear Avançada / Densa)", "darkorange", "Núcleo endurecido. Alto risco de perda endotelial."

    # 8. Exibição do Laudo Clínico Imediato na tela (Formato Limpo e Seguro)
    st.markdown("---")
    st.markdown("### 📊 Laudo Computacional")
    st.subheader(laudo)
    
    # Caixa de alerta colorida automática para a conduta médica
    if cor in ["purple", "red", "darkorange", "orange"]:
        st.warning(f"⚠️ **Conduta Cirúrgica:** {conduta}")
    else:
        st.success(f"✅ **Conduta Cirúrgica:** {conduta}")
