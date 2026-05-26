import streamlit as st
import cv2
import numpy as np

# 1. Configuração da Identidade Visual Médica do Software
st.set_page_config(
    page_title="NucleoClass Híbrido", 
    layout="centered", 
    page_icon="👁️"
)
st.title("👁️ NucleoClass - Sistema de Classificação Digital")
st.subheader("Densitometria Óptica Inteligente com Ajuste de Confirmação Visual")
st.caption("Versão Homologada e Estabilizada para o Protocolo do TCC")
st.markdown("---")

# BARRA LATERAL: Ajuste Fino de Calibração (Filtro anti-deslocamento)
st.sidebar.title("🛠️ Ajuste de Alinhamento")
st.sidebar.write("Se a fenda estiver deslocada na foto, use os controles abaixo para alinhar a área de leitura:")
deslocar_x = st.sidebar.slider("↔️ Ajuste Horizontal", min_value=-25, max_value=25, value=0, step=1)
deslocar_y = st.sidebar.slider("↕️ Ajuste Vertical", min_value=-20, max_value=20, value=0, step=1)

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda Fina Vertical):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. ENQUADRAMENTO DA ROI VERTICAL (Centro padrão + Ajuste fino da barra lateral)
    # Porcentagem padrão de fábrica: 40% a 60% na vertical e 46% a 54% na largura
    centro_x_padrao = 50 + deslocar_x
    centro_y_padrao = 50 + deslocar_y
    
    ymin = int(altura * ((centro_y_padrao - 10) / 100.0))
    ymax = int(altura * ((centro_y_padrao + 10) / 100.0))
    xmin = int(largura * ((centro_x_padrao - 4) / 100.0))
    xmax = int(largura * ((centro_x_padrao + 4) / 100.0))
    
    # Garantia de limites da imagem para evitar quebra do código
    xmin, xmax = max(0, xmin), min(largura, xmax)
    ymin, ymax = max(0, ymin), min(altura, ymax)
    
    # 5. PROCESSAMENTO DIGITAL DE SINAIS (Espaço HSV e RGB)
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
    
    # 6. EXIBIÇÃO VISUAL DO ENQUADRAMENTO DA IA
    img_viz = img.copy()
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Área Verde Selecionada para Análise do Núcleo", use_container_width=True)
    
    # 7. MOTOR DE DECISÃO AUTOMÁTICO COM RÉGUA CALIBRADA PELO SMARTPHONE
    
    # REGRA DA CATARATA BRANCA (G5): Saturação muito baixa em relação ao brilho adaptado
    if media_s < 45.0 and media_v > 100.0:
        laudo, cor = "G5 - Variante Catarata Branca / Total Intumescente", "red"
        conduta = "Opacificação total cortical. Alto risco de hipertensão intralenticular (Sinal da Bandeira Argentina). Realizar descompressão prévia com agulha fina antes da capsulorréxis. Usar Azul de Tripano obrigatório."
        faco_param = {"Torsional (Ozil)": "0% (Usar apenas I/A inicial)", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
    
    # REGRA DA CATARATA RUBRA (G6): Razão de cores alta mesmo em ambiente subexposto
    elif razao_vermelho_azul > 2.8 and media_v > 50.0:
        laudo, cor = "G6 - Variante Catarata Rubra / Brunescente Ultra-Densa", "purple"
        conduta = "Dureza máxima (rocha). Absorção cromática severa. Exige proteção endotelial máxima (Soft-Shell rígido) e parâmetros de alta energia torsional (Centurion Ozil 100% Contínuo)."
        faco_param = {"Torsional (Ozil)": "100% Contínuo", "Faco Longitudinal": "20-30% em Pulso", "Vácuo Máximo": "450-500 mmHg", "Fluxo de Aspiração": "40-45 cc/min", "IOP Alvo": "80 mmHg"}
    
    # ESCALA PROGRESSIVA NUCLEAR TÍPICA RECALIBRADA (G0 a G4)
    else:
        if media_v <= 50.0:
            laudo, cor = "G0 - Cristalino Transparente / Catarata Nuclear Incipiente", "green"
            conduta = "Parâmetros mínimos de energia. Cristalino gelatinoso e macio. Priorizar aspiração mecânica pura."
            faco_param = {"Torsional (Ozil)": "0%", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
        elif media_v <= 100.0:
            laudo, cor = "G1 - Grau I (Catarata Nuclear Inicial)", "green"
            conduta = "Fragmentação fácil. Baixa densidade nuclear. Parâmetros cirúrgicos conservadores de baixa energia."
            faco_param = {"Torsional (Ozil)": "20% Burst", "Faco Longitudinal": "0% Linear", "Vácuo Máximo": "350 mmHg", "Fluxo de Aspiração": "32 cc/min", "IOP Alvo": "60 mmHg"}
        elif media_v <= 145.0:
            laudo, cor = "G2 - Grau II (Catarata Nuclear Moderada-Leve)", "blue"
            conduta = "Densidade moderada padrão. Fragmentação mecânica fácil. Procedimento convencional estável do serviço."
            faco_param = {"Torsional (Ozil)": "40% Burst/Pulse", "Faco Longitudinal": "0-5% Linear", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "35 cc/min", "IOP Alvo": "65 mmHg"}
        elif media_v <= 190.0:
            laudo, cor = "G3 - Grau III (Catarata Nuclear Moderada-Avançada)", "orange"
            conduta = "Núcleo denso. Obrigatoriedade de técnicas mecânicas de fratura (Faco-Chop ou Quick Chop) para poupar energia ultrassônica total (CDE)."
            faco_param = {"Torsional (Ozil)": "60% Linear", "Faco Longitudinal": "10% Mili-burst", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "38 cc/min", "IOP Alvo": "70 mmHg"}
        else:
            laudo, cor = "G4 - Grau IV (Catarata Nuclear Avançada / Densa Típica)", "darkorange"
            conduta = "Cristalino altamente endurecido. Alto risco de perda endotelial e estresse zonular. Injetar viscoelástico dispersivo (Viscoat) repetidas vezes durante o procedimento."
            faco_param = {"Torsional (Ozil)": "80-100% Contínuo", "Faco Longitudinal": "15-20% Mili-burst", "Vácuo Máximo": "450 mmHg", "Fluxo de Aspiração": "40 cc/min", "IOP Alvo": "75 mmHg"}

    # 10. Entrega do Laudo e Métricas na Tela
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
