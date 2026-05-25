import streamlit as st
import cv2
import numpy as np

# 1. Configuração da Identidade Visual Médica do Software
st.set_page_config(
    page_title="CataractApp NucleoClass", 
    layout="centered", 
    page_icon="https://flaticon.com"
)
st.title("👁️ Novo Sistema Digital Automatizado de Classificação de Catarata")
st.subheader("Classificação Inteligente por Segmentação de Contorno do Núcleo")
st.caption("Versão Homologada: Segmentação Densitométrica Inteligente por Canal V (HSV)")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda ou Feixe Aberto):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. MARGEM DE SEGURANÇA EXPANDIDA (Dá espaço para a IA varrer o centro do olho)
    ymin, ymax = int(altura * 0.30), int(altura * 0.70)  
    xmin, xmax = int(largura * 0.35), int(largura * 0.65) 
    roi_bgr = img[ymin:ymax, xmin:xmax]
    
    # 5. INTELIGÊNCIA ARTIFICIAL: FILTRO BINÁRIO POR DENSIDADE DE BRILHO (Substitui o Canny)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_hsv = img_hsv[ymin:ymax, xmin:xmax]
    roi_v = roi_hsv[:, :, 2] # Isola a luminosidade pura
    
    # Suaviza a imagem e aplica uma máscara binária: isola o bloco de luz eliminando o fundo escuro do olho
    roi_blur = cv2.GaussianBlur(roi_v, (5, 5), 0)
    _, mascara_binaria = cv2.threshold(roi_blur, 45, 255, cv2.THRESH_BINARY)
    
    # Encontra os contornos desse bloco de luz real
    contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Máscara final de captura de pixels
    mascara_final = np.zeros(roi_v.shape, dtype=np.uint8)
    img_viz = img.copy()
    
    # A IA valida e seleciona o contorno real do feixe de luz (descarta ruídos e pontinhos)
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > 200]
    
    if contornos_validos:
        maior_contorno = max(contornos_validos, key=cv2.contourArea)
        cv2.drawContours(mascara_final, [maior_contorno], -1, 255, thickness=cv2.FILLED)
        
        # Desenha a linha verde anatômica real adaptada ao formato do feixe na tela do médico
        contorno_ajustado = maior_contorno + np.array([xmin, ymin])
        cv2.drawContours(img_viz, [contorno_ajustado], -1, (0, 255, 0), 4)
        caption_imagem = "Fenda Anatômica do Núcleo Detectada e Segmentada com Sucesso"
    else:
        # Contingência absoluta caso a foto esteja escura demais
        cv2.rectangle(mascara_final, (int(roi_v.shape[1]*0.4), int(roi_v.shape[0]*0.3)), (int(roi_v.shape[1]*0.6), int(roi_v.shape[0]*0.7)), 255, -1)
        cv2.rectangle(img_viz, (int(largura * 0.46), int(altura * 0.40)), (int(largura * 0.54), int(altura * 0.60)), (0, 255, 0), 4)
        caption_imagem = "Modo de Contingência: Aplicado Retorcido Central de Segurança"

    # Exibe a imagem com o contorno ajustado na tela
    st.image(img_viz, channels="BGR", caption=caption_imagem, use_container_width=True)
    
    # 6. EXTRAÇÃO MULTIDIMENSIONAL DE MÉTRICAS FILTRADAS PELA GEOMETRIA DA IA
    media_s = float(cv2.mean(roi_hsv[:, :, 1], mask=mascara_final)[0]) # Saturação
    media_v = float(cv2.mean(roi_hsv[:, :, 2], mask=mascara_final)[0]) # Luminosidade/Brilho V
    
    # Extração dos canais RGB originais para cálculo da razão dentro do contorno
    media_r = float(cv2.mean(roi_bgr[:, :, 2], mask=mascara_final)[0])
    media_b = float(cv2.mean(roi_bgr[:, :, 0], mask=mascara_final)[0])
    razao_vermelho_azul = media_r / (media_b + 0.001)
    
    # 7. MOTOR DE DECISÃO INTELIGENTE RECALIBRADO PARA ÁREA INTEGRAL
    
    # REGRA DA CATARATA BRANCA (G5): Saturação de cor baixa (gesso leitoso) + Brilho expressivo
    if media_s < 45.0 and media_v > 115.0:
        laudo = "G5 - Variante Catarata Branca / Total Intumescente"
        cor = "red"
        conduta = "Opacificação total cortical. Alto risco de hipertensão intralenticular (Sinal da Bandeira Argentina). Realizar descompressão prévia com agulha fina antes da capsulorréxis. Usar Azul de Tripano obrigatório."
        faco_param = {"Torsional (Ozil)": "0% (Usar apenas I/A inicial)", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
    
    # REGRA DA CATARATA RUBRA (G6): Razão de vermelho/azul alta (tom de tijolo profundo/marrom)
    elif razao_vermelho_azul > 3.0 and media_v > 60.0:
        laudo = "G6 - Variante Catarata Rubra / Brunescente Ultra-Densa"
        cor = "purple"
        conduta = "Dureza máxima (rocha). Absorção cromática severa. Exige proteção endotelial máxima (Soft-Shell rígido) e parâmetros de alta energia torsional (Centurion Ozil 100% Contínuo)."
        faco_param = {"Torsional (Ozil)": "100% Contínuo", "Faco Longitudinal": "20-30% em Pulso", "Vácuo Máximo": "450-500 mmHg", "Fluxo de Aspiração": "40-45 cc/min", "IOP Alvo": "80 mmHg"}
    
    # ESCALA PROGRESSIVA NUCLEAR TÍPICA (G0 a G4) - Baseada no Brilho V concentrado do contorno real
    else:
        if media_v <= 55.0:
            laudo = "G0 - Cristalino Transparente / Catarata Nuclear Incipiente"
            cor = "green"
            conduta = "Parâmetros mínimos de energia. Cristalino gelatinoso e macio. Priorizar aspiração mecânica pura."
            faco_param = {"Torsional (Ozil)": "0%", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
        elif media_v <= 110.0:
            laudo = "G1 - Grau I (Catarata Nuclear Inicial)"
            cor = "green"
            conduta = "Fragmentação fácil. Baixa densidade nuclear. Parâmetros cirúrgicos conservadores de baixa energia."
            faco_param = {"Torsional (Ozil)": "20% Burst", "Faco Longitudinal": "0% Linear", "Vácuo Máximo": "350 mmHg", "Fluxo de Aspiração": "32 cc/min", "IOP Alvo": "60 mmHg"}
        elif media_v <= 150.0:
            laudo = "G2 - Grau II (Catarata Nuclear Moderada-Leve)"
            cor = "blue"
            conduta = "Densidade moderada padrão. Fragmentação mecânica fácil. Procedimento convencional estável do serviço."
            faco_param = {"Torsional (Ozil)": "40% Burst/Pulse", "Faco Longitudinal": "0-5% Linear", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "35 cc/min", "IOP Alvo": "65 mmHg"}
        elif media_v <= 195.0:
            laudo = "G3 - Grau III (Catarata Nuclear Moderada-Avançada)"
            cor = "orange"
            conduta = "Núcleo denso. Obrigatoriedade de técnicas mecânicas de fratura (Faco-Chop ou Quick Chop) para poupar energia ultrassônica total (CDE)."
            faco_param = {"Torsional (Ozil)": "60% Linear", "Faco Longitudinal": "10% Mili-burst", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "38 cc/min", "IOP Alvo": "70 mmHg"}
        else:
            laudo = "G4 - Grau IV (Catarata Nuclear Avançada / Densa Típica)"
            cor = "darkorange"
            conduta = "Cristalino altamente endurecido. Alto risco de perda endotelial e estresse zonular. Injetar viscoelástico dispersivo (Viscoat) repetidas vezes durante o procedimento."
            faco_param = {"Torsional (Ozil)": "80-100% Contínuo", "Faco Longitudinal": "15-20% Mili-burst", "Vácuo Máximo": "450 mmHg", "Fluxo de Aspiração": "40 cc/min", "IOP Alvo": "75 mmHg"}

    # 8. Entrega do Laudo na Tela
    st.markdown("---")
    st.markdown("### 📊 Laudo Computacional Automatizado")
    st.subheader(laudo)
    
    # Exibição das métricas analíticas em Tabela Científica
    st.markdown("#### 🔬 Matriz de Parâmetros Ópticos do Núcleo")
    dados_metricas = {
        "Métrica Analisada pelo Segmentador": [
            "Brilho Médio do Núcleo Segmentado (Canal V)", 
            "Saturação de Cor Interna (Canal S)", 
            "Razão Cromática Pura do Núcleo (Vermelho / Azul)"
        ],
        "Valor Numérico Extraído": [
            f"{media_v:.1f}", 
            f"{media_s:.1f}", 
            f"{razao_vermelho_azul:.2f}"
        ]
    }
    st.table(dados_metricas)
    
    if cor in ["purple", "red", "darkorange", "orange"]:
        st.warning(f"⚠️ **Diretriz Cirúrgica:** {conduta}")
    else:
        st.success(f"✅ **Diretriz Cirúrgica:** {conduta}")
        
    # 9. Painel de Parâmetros Alcon Centurion
    st.markdown("### ⚙️ Programação Sugerida para Alcon Centurion")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Energia Torsional (Ozil)", faco_param["Torsional (Ozil)"])
        st.metric("Vácuo Máximo", faco_param["Vácuo Máximo"])
        st.metric("Pressão Intraocular (IOP)", faco_param["IOP Alvo"])
    with col2:
        st.metric("Faco Longitudinal", faco_param["Faco Longitudinal"])
        st.metric("Fluxo de Aspiração", faco_param["Fluxo de Aspiração"])
