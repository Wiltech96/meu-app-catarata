import streamlit as st
import cv2
import numpy as np

# 1. Configuração do aplicativo profissional
st.set_page_config(page_title="NucleoClass HSV", layout="centered", page_icon="👁️")
st.title("👁️ NucleoClass - Sistema Densitométrico Automático")
st.subheader("Classificação Avançada (G0-G6) por Análise de Matiz e Luminosidade (HSV)")
st.markdown("---")

arquivo = st.file_uploader("Insira a foto da lâmpada de fenda (Aceita Modo Automático):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # Ler o arquivo de imagem do smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # RECORTE PERCENTUAL DINÂMICO (Área central do núcleo)
    ymin, ymax = int(altura * 0.40), int(altura * 0.60)
    xmin, xmax = int(largura * 0.40), int(largura * 0.70)
    
    # CONVERSÃO PARA O ESPAÇO DE CORES HSV (O segredo contra fotos automáticas)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_hsv = img_hsv[ymin:ymax, xmin:xmax]
    
    # Extração das médias dos três canais na área do núcleo
    media_h = np.mean(roi_hsv[:, :, 0])  # Matiz (Cor Pura)
    media_s = np.mean(roi_hsv[:, :, 1])  # Saturação (Vivacidade da cor)
    media_v = np.mean(roi_hsv[:, :, 2])  # Valor (Brilho Puro)
    
    # Desenha o retângulo visual de conferência na imagem original
    img_viz = img.copy()
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Área Processada pelo Sensor HSV", use_container_width=True)
    
    # MOTOR DE DECISÃO MULTIDIMENSIONAL HSV
    
    # EXCEÇÃO G5: Catarata Branca Total (Saturação muito baixa + Brilho alto)
    if media_s < 45.0 and media_v > 120.0:
        laudo = "G5 - Variante Catarata Branca / Total Intumescente"
        cor, conduta = "red", "Opacificação cortical total. Alto risco de hipertensão intralenticular. Usar Azul de Tripano para capsulorréxis."
    
    # EXCEÇÃO G6: Catarata Rubra / Brunescente (Matiz deslocado para Vermelho/Marrom profundo)
    elif (media_h < 18 or media_h > 165) and media_v > 50.0 and media_s > 60.0:
        laudo = "G6 - Variante Catarata Rubra / Brunescente Ultra-Densa"
        cor, conduta = "purple", "Alta densidade nuclear (rocha). Exige proteção endotelial máxima (Soft-Shell rígido) e parâmetros de alta energia torsional (Centurion 100%)."
        
    # PROGRESSÃO NUCLEAR TÍPICA DO SERVIÇO (Baseada no brilho puro do canal V)
    else:
        if media_v <= 55.0:
            laudo = "G0 - Cristalino Transparente / Catarata Nuclear Incipiente"
            cor, conduta = "green", "Parâmetros mínimos de energia. Cristalino macio. Priorizar aspiração mecânica pura."
        elif media_v <= 105.0:
            laudo = "G1 - Grau I (Catarata Nuclear Inicial)"
            cor, conduta = "limegreen", "Fragmentação mecânica fácil. Baixa densidade. Parâmetros cirúrgicos conservadores."
        elif media_v <= 155.0:
            laudo = "G2 - Grau II (Catarata Nuclear Moderada-Leve)"
            cor, conduta = "blue", "Densidade moderada padrão do serviço. Procedimento convencional estável."
        elif media_v <= 200.0:
            laudo = "G3 - Grau III (Catarata Nuclear Moderada-Avançada)"
            cor, conduta = "orange", "Núcleo denso. Obrigatoriedade de técnicas mecânicas de fratura (Faco-Chop) para reduzir o tempo de ultrassom."
        else:
            laudo = "G4 - Grau IV (Catarata Nuclear Avançada / Densa Típica)"
            cor, conduta = "darkorange", "Cristalino altamente endurecido. Alto risco de perda endotelial. Injetar viscoelástico dispersivo repetidas vezes."

    # Exibição do Laudo Clínico Imediato na tela
    st.markdown("---")
    st.markdown("### 📊 Laudo Densitométrico Computacional")
    st.subheader(laudo)
    
    # Detalhes técnicos extras para enriquecer seu TCC
    st.write(f"🔬 *Métricas Físicas do Núcleo: Matiz(H): {media_h:.1f} | Saturação(S): {media_s:.1f} | Luminosidade(V): {media_v:.1f}*")
    
    if cor in ["purple", "red", "darkorange", "orange"]:
        st.warning(f"⚠️ **Diretriz Cirúrgica Sugerida:** {conduta}")
    else:
        st.success(f"✅ **Diretriz Cirúrgica Sugerida:** {conduta}")
