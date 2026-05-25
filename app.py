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
st.caption("Versão Homologada: Algoritmo Adaptativo com Filtro de Luminância Digital (ITU-R BT.601)")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda ou Feixe Aberto):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. RECORTE ANATÔMICO PADRÃO (Área Central da Pupila)
    ymin, ymax = int(altura * 0.35), int(altura * 0.65)  
    xmin, xmax = int(largura * 0.35), int(largura * 0.65) 
    
    # 5. PROCESSAMENTO DIGITAL DE SINAIS (RGB, HSV e LUMINÂNCIA PURA)
    canal_red = img[:, :, 2]
    canal_green = img[:, :, 1]
    canal_blue = img[:, :, 0]
    
    # Extração das médias RGB na Área de Interesse (ROI)
    media_r = float(np.mean(canal_red[ymin:ymax, xmin:xmax]))
    media_g = float(np.mean(canal_green[ymin:ymax, xmin:xmax]))
    media_b = float(np.mean(canal_blue[ymin:ymax, xmin:xmax]))
    
    # CÁLCULO DA LUMINÂNCIA REAL COMBINADA (Fórmula Óptica Perceptual BT.601)
    luminancia_y = (0.299 * media_r) + (0.587 * media_g) + (0.114 * media_b)
    
    # Espaço HSV para cálculo de saturação e suporte de brilho V
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    media_s = float(np.mean(img_hsv[ymin:ymax, xmin:xmax, 1])) # Saturação
    media_v = float(np.mean(img_hsv[ymin:ymax, xmin:xmax, 2])) # Brilho V
    
    # Cálculo da razão cromática adaptativa Vermelho/Azul (Extremo G6)
    razao_vermelho_azul = media_r / (media_b + 0.001)
    
    # 6. Desenha o retângulo visual na imagem para conferência do médico
    img_viz = img.copy()
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Área de Amostragem Analisada pelo Sensor", use_container_width=True)
    
    # 7. MOTOR DE DECISÃO INTELIGENTE ATUALIZADO (Filtros de Luminância e Razão)
    
    # TRAVA DEFINITIVA CATARATA BRANCA (G5): Saturação baixa (gesso) OU Luminância total estourada (leitoso)
    if (media_s < 40.0 and luminancia_y > 110.0) or (luminancia_y >= 150.0 and razao_vermelho_azul < 2.0):
        laudo = "G5 - Variante Catarata Branca / Total Intumescente"
        cor = "red"
        conduta = "Opacificação total cortical e estouro de reflexão. Alto risco de hipertensão intralenticular (Sinal da Bandeira Argentina). Realizar descompressão prévia com agulha fina antes da capsulorréxis. Usar Azul de Tripano obrigatório."
        faco_param = {"Torsional (Ozil)": "0% (Usar apenas I/A inicial)", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
    
    # TRAVA CATARATA RUBRA (G6): Razão de vermelho/azul alta (tom de tijolo/marrom escuro profundo)
    elif razao_vermelho_azul > 3.0 and media_v > 65.0:
        laudo = "G6 - Variante Catarata Rubra / Brunescente Ultra-Densa"
        cor = "purple"
        conduta = "Dureza máxima (rocha). Absorção cromática severa. Exige proteção endotelial máxima (Soft-Shell rígido) e parâmetros de alta energia torsional (Centurion Ozil 100% Contínuo)."
        faco_param = {"Torsional (Ozil)": "100% Contínuo", "Faco Longitudinal": "20-30% em Pulso", "Vácuo Máximo": "450-500 mmHg", "Fluxo de Aspiração": "40-45 cc/min", "IOP Alvo": "80 mmHg"}
    
    # ESCALA PROGRESSIVA NUCLEAR TÍPICA (G0 a G4) - Baseada no Brilho V Estabilizado
    else:
        if media_v <= 55.0:
            laudo = "G0 - Cristalino Transparente / Catarata Nuclear Incipiente"
            cor = "green"
            conduta = "Parâmetros mínimos de energia. Cristalino gelatinoso e macio. Priorizar aspiração mecânica pura."
            faco_param = {"Torsional (Ozil)": "0%", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
        elif media_v <= 105.0:
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
    st.markdown("### 📊 Laudo Computacional")
    st.subheader(laudo)
    
    # Painel de Auditoria Densitométrica (Essencial para o TCC)
    st.markdown("#### 🔬 Métricas do Núcleo")
    dados_metricas = {
        "Métrica Analisada": ["Luminância Perceptual (Y)", "Brilho Puro (Canal V)", "Saturação de Cor (Canal S)", "Razão Cromática (R/A)"],
        "Valor Extraído": [f"{luminancia_y:.1f}", f"{media_v:.1f}", f"{media_s:.1f}", f"{razao_vermelho_azul:.2f}"]
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
