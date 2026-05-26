import streamlit as st
import cv2
import numpy as np

# 1. Configuração da Identidade Visual Médica do Software
st.set_page_config(
    page_title="CataractApp Dataset Tool", 
    layout="centered", 
    page_icon="🔬"
)
st.title("🔬 NucleoClass - Ferramenta de Marcação e Densitometria")
st.subheader("Módulo de Delimitação Manual de ROI para Treinamento de IA")
st.caption("Foco Metodológico: Criação de Dataset e Padronização Ambulatorial")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda Fina Vertical):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    st.markdown("### 🛠️ Ajuste Anatômico do Núcleo")
    st.write("Use os controles abaixo para alinhar o retângulo verde perfeitamente sobre o núcleo do cristalino do paciente:")
    
    # 4. SLIDERS INTELIGENTES: Permitem ao médico delimitar manualmente a área correta
    col_pos, col_dim = st.columns(2)
    with col_pos:
        pos_x = st.slider("↔️ Posição Horizontal (Centro)", min_value=10, max_value=90, value=50, step=1)
        pos_y = st.slider("↕️ Posição Vertical (Centro)", min_value=10, max_value=90, value=50, step=1)
    with col_dim:
        largura_roi = st.slider("📏 Largura do Retângulo", min_value=4, max_value=30, value=10, step=1)
        altura_roi = st.slider("📐 Altura do Retângulo", min_value=10, max_value=50, value=25, step=1)
        
    # Conversão das porcentagens dos sliders para pixels reais da imagem
    centro_x = int(largura * (pos_x / 100.0))
    centro_y = int(altura * (pos_y / 100.0))
    raio_x = int(largura * (largura_roi / 200.0))
    raio_y = int(altura * (altura_roi / 200.0))
    
    # Coordenadas finais da ROI delimitada pelo médico
    xmin, xmax = max(0, centro_x - raio_x), min(largura, centro_x + raio_x)
    ymin, ymax = max(0, centro_y - raio_y), min(altura, centro_y + raio_y)
    
    # Evitar ROIs zeradas
    if xmax <= xmin: xmax = xmin + 5
    if ymax <= ymin: ymax = ymin + 5
    
    # 5. Desenha o retângulo visual na imagem baseado nos sliders
    img_viz = img.copy()
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Conferência Clínica: Garanta que a área verde cubra apenas o núcleo", use_container_width=True)
    
    # 6. PROCESSAMENTO DIGITAL DE SINAIS (Espaço HSV e RGB dentro da área escolhida)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_hsv = img_hsv[ymin:ymax, xmin:xmax]
    
    media_s = float(np.mean(roi_hsv[:, :, 1])) # Saturação
    media_v = float(np.mean(roi_hsv[:, :, 2])) # Luminosidade
    
    canal_red = img[ymin:ymax, xmin:xmax, 2]
    canal_blue = img[ymin:ymax, xmin:xmax, 0]
    media_r = float(np.mean(canal_red))
    media_b = float(np.mean(canal_blue))
    razao_vermelho_azul = media_r / (media_b + 0.001)
    
    # 7. MOTOR DE DECISÃO INTELIGENTE (Escala G0-G6 Travada)
    if media_s < 45.0 and media_v > 115.0:
        laudo, cor = "G5 - Variante Catarata Branca / Total Intumescente", "red"
        conduta = "Opacificação total cortical. Alto risco de hipertensão intralenticular (Sinal da Bandeira Argentina). Realizar descompressão prévia com agulha fina antes da capsulorréxis. Usar Azul de Tripano obrigatório."
        faco_param = {"Torsional (Ozil)": "0% (Usar apenas I/A inicial)", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
    elif razao_vermelho_azul > 3.2 and media_v > 60.0:
        laudo, cor = "G6 - Variante Catarata Rubra / Brunescente Ultra-Densa", "purple"
        conduta = "Dureza máxima (rocha). Absorção cromática severa. Exige proteção endotelial máxima (Soft-Shell rígido) e parâmetros de alta energia torsional (Centurion Ozil 100% Contínuo)."
        faco_param = {"Torsional (Ozil)": "100% Contínuo", "Faco Longitudinal": "20-30% em Pulso", "Vácuo Máximo": "450-500 mmHg", "Fluxo de Aspiração": "40-45 cc/min", "IOP Alvo": "80 mmHg"}
    else:
        if media_v <= 50.0:
            laudo, cor = "G0 - Cristalino Transparente / Catarata Nuclear Incipiente", "green"
            conduta = "Parâmetros mínimos de energia. Cristalino gelatinoso e macio. Priorizar aspiração mecânica pura."
            faco_param = {"Torsional (Ozil)": "0%", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
        elif media_v <= 105.0:
            laudo, cor = "G1 - Grau I (Catarata Nuclear Inicial)", "green"
            conduta = "Fragmentação fácil. Baixa densidade nuclear. Parâmetros cirúrgicos conservadores de baixa energia."
            faco_param = {"Torsional (Ozil)": "20% Burst", "Faco Longitudinal": "0% Linear", "Vácuo Máximo": "350 mmHg", "Fluxo de Aspiração": "32 cc/min", "IOP Alvo": "60 mmHg"}
        elif media_v <= 150.0:
            laudo, cor = "G2 - Grau II (Catarata Nuclear Moderada-Leve)", "blue"
            conduta = "Densidade moderada padrão. Fragmentação mecânica fácil. Procedimento convencional estável do serviço."
            faco_param = {"Torsional (Ozil)": "40% Burst/Pulse", "Faco Longitudinal": "0-5% Linear", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "35 cc/min", "IOP Alvo": "65 mmHg"}
        elif media_v <= 195.0:
            laudo, cor = "G3 - Grau III (Catarata Nuclear Moderada-Avançada)", "orange"
            conduta = "Núcleo denso. Obrigatoriedade de técnicas mecânicas de fratura (Faco-Chop ou Quick Chop) para poupar energia ultrassônica total (CDE)."
            faco_param = {"Torsional (Ozil)": "60% Linear", "Faco Longitudinal": "10% Mili-burst", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "38 cc/min", "IOP Alvo": "70 mmHg"}
        else:
            laudo, cor = "G4 - Grau IV (Catarata Nuclear Avançada / Densa Típica)", "darkorange"
            conduta = "Cristalino altamente endurecido. Alto risco de perda endotelial e estresse zonular. Injetar viscoelástico dispersivo (Viscoat) repetidas vezes durante o procedimento."
            faco_param = {"Torsional (Ozil)": "80-100% Contínuo", "Faco Longitudinal": "15-20% Mili-burst", "Vácuo Máximo": "450 mmHg", "Fluxo de Aspiração": "40 cc/min", "IOP Alvo": "75 mmHg"}

    # 8. Entrega do Laudo e Dados na Tela
    st.markdown("---")
    st.markdown("### 📊 Laudo Gerado para a Região Marcada")
    st.subheader(laudo)
    
    st.markdown("#### 🔬 Coordenadas e Métricas do Dataset (Salve estes dados)")
    dados_metricas = {
        "Parâmetro Clínico/Técnico": ["Brilho Localizado (V)", "Saturação Localizada (S)", "Razão R/A", "Coordenadas da Caixa (Xmin, Xmax, Ymin, Ymax)"],
        "Valor Extraído": [f"{media_v:.1f}", f"{media_s:.1f}", f"{razao_vermelho_azul:.2f}", f"[{xmin}, {xmax}, {ymin}, {ymax}]"]
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
