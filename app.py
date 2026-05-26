import streamlit as st
import cv2
import numpy as np

# 1. Configuração visual do aplicativo (Seu padrão original)
st.set_page_config(page_title="CataractApp G0-G6", layout="centered", page_icon="👁️")
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
    
    # 4. RECORTE PERCENTUAL DINÂMICO ORIGINAL (A régua fixa que você calibrou)
    ymin, ymax = int(altura * 0.40), int(altura * 0.60)
    xmin, xmax = int(largura * 0.40), int(largura * 0.70)
    
    # 5. FILTRAGEM HSV COMPATÍVEL COM O MODO AUTOMÁTICO
    # Converte a imagem para HSV apenas para extrair a luminosidade limpa (Canal V)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    canal_v = img_hsv[:, :, 2]
    
    # Mantém os canais originais para a checagem da Catarata Rubra e Branca
    canal_red = img[:, :, 2]
    canal_blue = img[:, :, 0]
    
    # Extração das médias estritamente dentro do seu retângulo fixo original
    media_v = np.mean(canal_v[ymin:ymax, xmin:xmax])
    media_vermelho = np.mean(canal_red[ymin:ymax, xmin:xmax])
    media_azul = np.mean(canal_blue[ymin:ymax, xmin:xmax])
    
    # 6. Desenha o retângulo verde original para conferência do médico
    img_viz = img.copy()
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Área de Leitura do Núcleo", use_container_width=True)
    
    # 7. MOTOR DE DECISÃO SEGURO (Régua clássica baseada no brilho filtrado)
    # Critério da Catarata Branca Total
    if media_azul >= 130.0 and media_vermelho > 100.0:
        laudo, cor, conduta = "G5 - Catarata Branca / Total Intumescente", "red", "Opacificação total. Usar Azul de Tripano para capsulorréxis."
    
    # Critério da Catarata Rubra/Brunescente (Proporção estável)
    elif media_azul < (media_vermelho / 2.8) and media_vermelho > 50.0:
        laudo, cor, conduta = "G6 - Catarata Rubra / Brunescente Ultra-Densa", "purple", "Alta densidade nuclear (rocha). Exige proteção endotelial máxima."
    
    # Escala Progressiva baseada na densidade do canal de brilho V
    else:
        if media_v <= 50.0:
            laudo, cor, conduta = "G0 - Cristalino Transparente / Incipiente", "green", "Parâmetros mínimos de energia."
        elif media_v <= 100.0:
            laudo, cor, conduta = "G1 - Grau I (Catarata Nuclear Inicial)", "green", "Fragmentação fácil. Energia padrão baixa."
        elif media_v <= 145.0:
            laudo, cor, conduta = "G2 - Grau II (Catarata Nuclear Moderada-Leve)", "blue", "Procedimento cirúrgico padrão estável."
        elif media_v <= 190.0:
            laudo, cor, conduta = "G3 - Grau III (Catarata Nuclear Moderada-Avançada)", "orange", "Núcleo denso. Considerar técnicas mecânicas (Faco-Chop)."
        else:
            laudo, cor, conduta = "G4 - Grau IV (Catarata Nuclear Avançada / Densa)", "darkorange", "Núcleo endurecido. Alto risco de perda endotelial."

    # 8. Exibição do Laudo Clínico Imediata na tela
    st.markdown("---")
    st.markdown("### 📊 Laudo Computacional")
    st.subheader(laudo)
    
    # Exibe discretamente os valores para te ajudar a anotar na planilha do TCC
    st.caption(f"Métricas de conferência -> Brilho (V): {media_v:.1f} | Vermelho: {media_vermelho:.1f} | Azul: {media_azul:.1f}")
    
    # Caixa de alerta colorida automática para a conduta médica
    if cor in ["purple", "red", "darkorange", "orange"]:
        st.warning(f"⚠️ **Conduta Cirúrgica:** {conduta}")
    else:
        st.success(f"✅ **Conduta Cirúrgica:** {conduta}")
