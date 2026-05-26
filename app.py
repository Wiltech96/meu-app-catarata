import streamlit as st
import cv2
import numpy as np

# 1. Configuração da Identidade Visual Médica do Software
st.set_page_config(
    page_title="CataractApp NucleoClass", 
    layout="centered", 
    page_icon="https://flaticon.com"
)
st.title("👁️ NucleoClass - Análise Densitométrica de Catarata")
st.subheader("Classificação Automatizada Ambulatorial (G0 a G6)")
st.caption("Versão Homologada: Reprodutibilidade Espacial por Amostragem de ROI Expandida")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda Fina Vertical):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. RESTRIÇÃO ANATÔMICA CENTRALIZADA (Foca a varredura na região útil da pupila)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_inicio = int(largura * 0.32)
    x_fim = int(largura * 0.68)
    y_inicio = int(altura * 0.35)
    y_fim = int(altura * 0.65)
    
    roi_busca = img_gray[y_inicio:y_fim, x_inicio:x_fim]
    
    # 5. ALGORITMO CENTRO DE MASSA (Encontra o coração geométrico da fenda / bola de beisebol)
    roi_blur = cv2.GaussianBlur(roi_busca, (5, 5), 0)
    _, thresh = cv2.threshold(roi_blur, 45, 255, cv2.THRESH_BINARY)
    
    momentos = cv2.moments(thresh)
    
    if momentos["m00"] != 0:
        centro_x_real = x_inicio + int(momentos["m10"] / momentos["m00"])
        centro_y_real = y_inicio + int(momentos["m01"] / momentos["m00"])
    else:
        centro_x_real = int(largura * 0.5)
        centro_y_real = int(altura * 0.5)
        
    # 6. DIMENSIONAMENTO DA ROI EXPANDIDA REPRODUTÍVEL
    # Expandido de forma calculada para diluir artefatos de reflexos pontuais e estabilizar a amostragem
    tamanho_x = int(largura * 0.08)  # Alargado para absorver a fenda total sem vazar para a esclera
    tamanho_y = int(altura * 0.15)  # Alongado para cobrir o miolo vertical do núcleo completo
    
    ymin, ymax = max(0, centro_y_real - tamanho_y), min(altura, centro_y_real + tamanho_y)
    xmin, xmax = max(0, centro_x_real - tamanho_x), min(largura, centro_x_real + tamanho_x)
    
    # 7. PROCESSAMENTO DIGITAL DE SINAIS (Lógica de Extração e Pesos Originais)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_hsv = img_hsv[ymin:ymax, xmin:xmax]
    
    media_h = float(np.mean(roi_hsv[:, :, 0])) # Matiz
    media_s = float(np.mean(roi_hsv[:, :, 1])) # Saturação
    media_v = float(np.mean(roi_hsv[:, :, 2])) # Luminosidade/Brilho Puro (Canal V)
    
    # Extração dos canais RGB originais na ROI Expandida
    canal_red = img[ymin:ymax, xmin:xmax, 2]
    canal_green = img[ymin:ymax, xmin:xmax, 1]
    canal_blue = img[ymin:ymax, xmin:xmax, 0]
    
    media_r = float(np.mean(canal_red))
    media_g = float(np.mean(canal_green))
    media_b = float(np.mean(canal_blue))
    
    # CÁLCULO DA LUMINÂNCIA PERCEPTUAL REAL (Fórmula Óptica Internacional ITU-R BT.601)
    luminancia_y = (0.299 * media_r) + (0.587 * media_g) + (0.114 * media_b)
    
    # Cálculo da razão cromática adaptativa Vermelho/Azul (Extremo G6)
    razao_vermelho_azul = media_r / (media_b + 0.001)
    
    # 8. EXIBIÇÃO VISUAL DO PROTÓTIPO NA TELA
    img_viz = img.copy()
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Área de Amostragem Ampliada para Diluição de Reflexos Especulares", use_container_width=True)
    
    # 9. MOTOR DE DECISÃO INTELIGENTE ORIGINAL (Régua Histórica Intacta e Estável)
    
    # REGRA DA CATARATA BRANCA (G5): Saturação de cor muito baixa (gesso leitoso) + Brilho expressivo
    if media_s < 45.0 and media_v > 115.0:
        laudo = "G5 - Variante Catarata Branca / Total Intumescente"
        cor = "red"
        conduta = "Opacificação total cortical. Alto risco de hipertensão intralenticular (Sinal da Bandeira Argentina). Realizar descompressão prévia com agulha fina antes da capsulorréxis. Usar Azul de Tripano obrigatório."
        faco_param = {"Torsional (Ozil)": "0% (Usar apenas I/A inicial)", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
    
    # REGRA DA CATARATA RUBRA (G6): Razão de vermelho/azul alta (tom de tijolo profundo/marrom)
    elif razao_vermelho_azul > 3.2 and media_v > 60.0:
        laudo = "G6 - Variante Catarata Rubra / Brunescente Ultra-Densa"
        cor = "purple"
        conduta = "Dureza máxima (rocha). Absorção cromática severa. Exige proteção endotelial máxima (Soft-Shell rígido) e parâmetros de alta energia torsional (Centurion Ozil 100% Contínuo)."
        faco_param = {"Torsional (Ozil)": "100% Contínuo", "Faco Longitudinal": "20-30% em Pulso", "Vácuo Máximo": "450-500 mmHg", "Fluxo de Aspiração": "40-45 cc/min", "IOP Alvo": "80 mmHg"}
    
    # ESCALA PROGRESSIVA NUCLEAR TÍPICA ORIGINAL (MANTIDA RIGOROSAMENTE INTACTA)
    else:
        if media_v <= 50.0:
            laudo = "G0 - Cristalino Transparente / Catarata Nuclear Incipiente"
            cor = "green"
            conduta = "Parâmetros mínimos de energia. Cristalino gelatinoso e macio. Priorizar aspiração mecânica pura."
            faco_param = {"Torsional (Ozil)": "0%", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
        elif media_v <= 100.0:
            laudo = "G1 - Grau I (Catarata Nuclear Inicial)"
            cor = "green"
            conduta = "Fragmentação fácil. Baixa densidade nuclear. Parâmetros cirúrgicos conservadores de baixa energia."
            faco_param = {"Torsional (Ozil)": "20% Burst", "Faco Longitudinal": "0% Linear", "Vácuo Máximo": "350 mmHg", "Fluxo de Aspiração": "32 cc/min", "IOP Alvo": "60 mmHg"}
        elif media_v <= 145.0:
            laudo = "G2 - Grau II (Catarata Nuclear Moderada-Leve)"
            cor = "blue"
            conduta = "Densidade moderada padrão. Fragmentação mecânica fácil. Procedimento convencional estável do serviço."
            faco_param = {"Torsional (Ozil)": "40% Burst/Pulse", "Faco Longitudinal": "0-5% Linear", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "35 cc/min", "IOP Alvo": "65 mmHg"}
        elif media_v <= 190.0:
            laudo = "G3 - Grau III (Catarata Nuclear Moderada-Avançada)"
            cor = "orange"
            conduta = "Núcleo denso. Obrigatoriedade de técnicas mecânicas de fratura (Faco-Chop ou Quick Chop) para poupar energia ultrassônica total (CDE)."
            faco_param = {"Torsional (Ozil)": "60% Linear", "Faco Longitudinal": "10% Mili-burst", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "38 cc/min", "IOP Alvo": "70 mmHg"}
        else:
            laudo = "G4 - Grau IV (Catarata Nuclear Avançada / Densa Típica)"
            cor = "darkorange"
            conduta = "Cristalino altamente endurecido. Alto risco de perda endotelial e estresse zonular. Injetar viscoelástico dispersivo (Viscoat) repetidas vezes durante o procedimento."
            faco_param = {"Torsional (Ozil)": "80-100% Contínuo", "Faco Longitudinal": "15-20% Mili-burst", "Vácuo Máximo": "450 mmHg", "Fluxo de Aspiração": "40 cc/min", "IOP Alvo": "75 mmHg"}

    # 10. Entrega do Laudo na Tela
    st.markdown("---")
    st.markdown("### 📊 Laudo Computacional")
    st.subheader(laudo)
    
    # Matriz Científica de Métricas do Núcleo Lenticular
    st.markdown("#### 🔬 Matriz de Parâmetros Ópticos")
    dados_metricas = {
        "Métrica Analisada": [
            "Luminância Perceptual (Y - Padrão BT.601)", 
            "Brilho Puro do Cristalino (Canal V)", 
            "Saturação Cromática (Canal S)", 
            "Razão Cromática Dinâmica (Vermelho / Azul)"
        ],
        "Valor Numérico Extraído": [
            f"{luminancia_y:.1f}", 
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
        
    # 11. Painel de Parâmetros Alcon Centurion
    st.markdown("### ⚙️ Programação Sugerida para Alcon Centurion")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Energia Torsional (Ozil)", faco_param["Torsional (Ozil)"])
        st.metric("Vácuo Máximo", faco_param["Vácuo Máximo"])
        st.metric("Pressão Intraocular (IOP)", faco_param["IOP Alvo"])
    with col2:
        st.metric("Faco Longitudinal", faco_param["Faco Longitudinal"])
        st.metric("Fluxo de Aspiração", faco_param["Fluxo de Aspiração"])
