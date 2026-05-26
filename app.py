import streamlit as st
import cv2
import numpy as np

# 1. Configuração da Identidade Visual Médica do Software
st.set_page_config(
    page_title="NucleoClass Auto", 
    layout="centered", 
    page_icon="👁️"
)
st.title("👁️ NucleoClass - Automação por Centro de Massa")
st.subheader("Classificação Automatizada por Segmentação Geométrica")
st.caption("Versão Final Homologada: Centralização Automática no Core da Fenda Lenticular")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda Fina Vertical):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. RESTRIÇÃO ANATÔMICA: Foca a busca na região central da pupila (descarta ruídos das bordas)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_inicio = int(largura * 0.30)
    x_fim = int(largura * 0.70)
    y_inicio = int(altura * 0.30)
    y_fim = int(altura * 0.70)
    
    roi_busca = img_gray[y_inicio:y_fim, x_inicio:x_fim]
    
    # 5. INTELIGÊNCIA ARTIFICIAL: CÁLCULO DO CENTRO DE MASSA (MOMENTOS DA IMAGEM)
    # Suaviza e binariza para isolar o corpo iluminado da fenda (a "bola de beisebol")
    roi_blur = cv2.GaussianBlur(roi_busca, (5, 5), 0)
    _, thresh = cv2.threshold(roi_blur, 50, 255, cv2.THRESH_BINARY)
    
    # Calcula os momentos matemáticos da forma geométrica
    momentos = cv2.moments(thresh)
    
    # Se encontrar o bloco iluminado, calcula as coordenadas do centro de gravidade
    if momentos["m00"] != 0:
        centro_x_relativo = int(momentos["m10"] / momentos["m00"])
        centro_y_relativo = int(momentos["m01"] / momentos["m00"])
        
        # Converte as coordenadas do centro de volta para o tamanho real da imagem inteira
        centro_x_real = x_inicio + centro_x_relativo
        centro_y_real = y_inicio + centro_y_relativo
    else:
        # Contingência padrão caso a imagem falhe na leitura
        centro_x_real = int(largura * 0.5)
        centro_y_real = int(altura * 0.5)
        
    # 6. ENCAIXA O RETÂNGULO VERDE AUTOMATICAMENTE NO CENTRO DA BOLA
    # Fixa um quadrado compacto que fica trancado dentro do núcleo profundo
    tamanho_quadrado_x = int(largura * 0.04) # Estreito para não vazar horizontalmente
    tamanho_quadrado_y = int(altura * 0.08)  # Altura ideal para o núcleo central
    
    ymin = max(0, centro_y_real - tamanho_quadrado_y)
    ymax = min(altura, centro_y_real + tamanho_quadrado_y)
    xmin = max(0, centro_x_real - tamanho_quadrado_x)
    xmax = min(largura, centro_x_real + tamanho_quadrado_x)
    
    # 7. PROCESSAMENTO DIGITAL DE SINAIS (Espaço HSV e RGB dentro da zona automática)
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
    
    # 8. EXIBIÇÃO VISUAL DO PROTÓTIPO NA TELA
    img_viz = img.copy()
    # Desenha um ponto vermelho discreto no centro de gravidade calculado pela IA
    cv2.circle(img_viz, (centro_x_real, centro_y_real), 6, (0, 0, 255), -1)
    # Desenha o retângulo verde travado no núcleo profundo
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Retângulo Focado Automaticamente no Centro de Gravidade do Núcleo", use_container_width=True)
    
    # 9. MOTOR DE DECISÃO AUTOMÁTICO COM RÉGUA CALIBRADA PELO SMARTPHONE
    if media_s < 35.0 and media_v > 40.0:
        laudo, cor = "G5 - Variante Catarata Branca / Total Intumescente", "red"
        conduta = "Opacificação total cortical. Alto risco de hipertensão intralenticular (Sinal da Bandeira Argentina). Realizar descompressão prévia com agulha fina antes da capsulorréxis. Usar Azul de Tripano obrigatório."
        faco_param = {"Torsional (Ozil)": "0% (Usar apenas I/A inicial)", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
    elif razao_vermelho_azul > 2.5 and media_v > 35.0:
        laudo, cor = "G6 - Variante Catarata Rubra / Brunescente Ultra-Densa", "purple"
        conduta = "Dureza máxima (rocha). Absorção cromática severa. Exige proteção endotelial máxima (Soft-Shell rígido) e parâmetros de alta energia torsional (Centurion Ozil 100% Contínuo)."
        faco_param = {"Torsional (Ozil)": "100% Contínuo", "Faco Longitudinal": "20-30% em Pulso", "Vácuo Máximo": "450-500 mmHg", "Fluxo de Aspiração": "40-45 cc/min", "IOP Alvo": "80 mmHg"}
    else:
        if media_v <= 20.0:
            laudo, cor = "G0 - Cristalino Transparente / Catarata Nuclear Incipiente", "green"
            conduta = "Parâmetros mínimos de energia. Cristalino gelatinoso e macio. Priorizar aspiração mecânica pura."
            faco_param = {"Torsional (Ozil)": "0%", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
        elif media_v <= 30.0:
            laudo, cor = "G1 - Grau I (Catarata Nuclear Inicial)", "green"
            conduta = "Fragmentação fácil. Baixa densidade nuclear. Parâmetros cirúrgicos conservadores de baixa energia."
            faco_param = {"Torsional (Ozil)": "20% Burst", "Faco Longitudinal": "0% Linear", "Vácuo Máximo": "350 mmHg", "Fluxo de Aspiração": "32 cc/min", "IOP Alvo": "60 mmHg"}
        elif media_v <= 42.0:
            laudo, cor = "G2 - Grau II (Catarata Nuclear Moderada-Leve)", "blue"
            conduta = "Densidade moderada padrão. Fragmentação mecânica fácil. Procedimento convencional estável do serviço."
            faco_param = {"Torsional (Ozil)": "40% Burst/Pulse", "Faco Longitudinal": "0-5% Linear", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "35 cc/min", "IOP Alvo": "65 mmHg"}
        elif media_v <= 52.0:
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
