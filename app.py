import streamlit as st
import cv2
import numpy as np

# 1. Configuração da Identidade Visual Médica do Software
st.set_page_config(
    page_title="CataractApp NucleoClass", 
    layout="centered", 
    page_icon="👁️"
)
st.title("👁️ Novo Sistema Digital Automatizado de Classificação de Catarata")
st.subheader("Classificação Inteligente por Segmentação de Contorno do Núcleo")
st.caption("Versão Final: Rastreamento Anatômico Intercapsular (Imune a Enquadramento)")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda Fina Vertical):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. ALGORITMO INTERCAPSULAR (Separação Córnea/Cristalino e Descarte do Fundo Preto)
    # Converte para tons de cinza para mapear a energia luminosa
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suavização para remover o ruído digital (noise) do sensor do celular
    blur = cv2.GaussianBlur(img_gray, (15, 15), 0)
    
    # Corta o "preto" em volta gerando uma máscara binarizada automática (Método de Otsu)
    # Isola tudo o que emite luz na fenda óptica
    _, mascara_luz = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Encontra os contornos de luz na imagem
    contornos, _ = cv2.findContours(mascara_luz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra contornos por tamanho para evitar reflexos espúrios
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > 2000]
    
    # Inicialização das coordenadas de segurança caso o rastreador falhe
    centro_x = int(largura * 0.5)
    centro_y = int(altura * 0.5)
    tamanho_roi = int(min(largura, altura) * 0.08)
    
    caption_imagem = "Modo de Segurança: Padrão Geométrico Aplicado"
    img_viz = img.copy()

    if len(contornos_validos) >= 1:
        # Ordena os blocos de luz da direita para a esquerda (coordenada X do contorno)
        # O bloco mais à direita na fenda é o cristalino (a córnea fica à esquerda)
        contornos_ordenados = sorted(contornos_validos, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        
        # Seleciona o contorno anatômico do cristalino
        cristalino_contorno = contornos_ordenados[0]
        x, y, w, h = cv2.boundingRect(cristalino_contorno)
        
        # Define as cápsulas anterior e posterior com base nos limites laterais do bloco
        capsula_anterior_x = x
        capsula_posterior_x = x + w
        
        # O miolo do núcleo fica exatamente no centro físico entre as duas cápsulas
        centro_x = int((capsula_anterior_x + capsula_posterior_x) / 2)
        centro_y = int(y + (h / 2))
        
        # Garante que a área de análise fique restrita ao interior do núcleo
        tamanho_roi = int(w * 0.3)  # Analisa os 30% centrais do bloco do cristalino
        
        # Desenha a linha anatômica do cristalino na tela do médico
        cv2.drawContours(img_viz, [cristalino_contorno], -1, (0, 255, 0), 3)
        # Desenha a mira central no miolo profundo
        cv2.circle(img_viz, (centro_x, centro_y), 8, (255, 0, 0), -1)
        caption_imagem = "Cristalino Isoloado do Fundo Preto. Mira Fixada entre as Cápsulas."

    # Define os limites finais de corte da ROI dinâmica
    ymin, ymax = max(0, centro_y - tamanho_roi), min(altura, centro_y + tamanho_roi)
    xmin, xmax = max(0, centro_x - tamanho_roi), min(largura, centro_x + tamanho_roi)
    
    # Desenha o quadrado da ROI na imagem de exibição
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 255), 4)
    st.image(img_viz, channels="BGR", caption=caption_imagem, use_container_width=True)
    
    # 5. PROCESSAMENTO AVANÇADO DO MIOLO ISOLADO (HSV)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_hsv = img_hsv[ymin:ymax, xmin:xmax]
    roi_bgr = img[ymin:ymax, xmin:xmax]
    
    # 6. EXTRAÇÃO MULTIDIMENSIONAL DE MÉTRICAS NO MIOLO
    media_h = float(np.mean(roi_hsv[:, :, 0])) # Matiz
    media_s = float(np.mean(roi_hsv[:, :, 1])) # Saturação
    media_v = float(np.mean(roi_hsv[:, :, 2])) # Luminosidade V (O nosso cinza estável)
    
    # Extração dos canais RGB originais para cálculo da razão dentro do miolo
    media_r = float(np.mean(roi_bgr[:, :, 2]))
    media_b = float(np.mean(roi_bgr[:, :, 0]))
    razao_vermelho_azul = media_r / (media_b + 0.001)
    
    # 7. MOTOR DE DECISÃO INTELIGENTE RECALIBRADO PARA ÁREA INTEGRAL DO NÚCLEO
    
    # REGRA DA CATARATA BRANCA (G5): Saturação de cor muito baixa (gesso leitoso) + Brilho expressivo
    if media_s < 45.0 and media_v > 115.0:
        laudo = "G5 - Variante Catarata Branca / Total Intumescente"
        cor = "red"
        conduta = "Opacificação total cortical. Alto risco de hipertensão intralenticular (Sinal da Bandeira Argentina). Realizar descompressão prévia com agulha fina antes da capsulorréxis. Usar Azul de Tripano obrigatório."
        faco_param = {"Torsional (Ozil)": "0% (Apenas Aspiração)", "Faco Longitudinal": "0% Linear", "Vácuo Máximo": "280 mmHg", "Fluxo de Aspiração": "28 cc/min", "IOP Alvo": "50 mmHg"}
    
    # REGRA DA CATARATA RUBRA (G6): Razão de vermelho/azul alta (tom de tijolo profundo/marrom)
    elif razao_vermelho_azul > 3.2 and media_v > 60.0:
        laudo = "G6 - Variante Catarata Rubra / Brunescente Ultra-Densa"
        cor = "purple"
        conduta = "Dureza máxima (rocha). Absorção cromática severa. Exige proteção endotelial máxima (Soft-Shell rígido com viscoelásticos dispersivo e coesivo) e parâmetros de alta energia torsional estável."
        faco_param = {"Torsional (Ozil)": "100% Contínuo", "Faco Longitudinal": "25% em Pulso", "Vácuo Máximo": "450-500 mmHg", "Fluxo de Aspiração": "42 cc/min", "IOP Alvo": "80 mmHg Active"}
    
    # ESCALA PROGRESSIVA NUCLEAR TÍPICA (G0 a G4) - Baseada no Brilho V concentrado do miolo real
    else:
        if media_v <= 55.0:
            laudo = "G0 - Cristalino Transparente / Catarata Nuclear Incipiente"
            cor = "green"
            conduta = "Parâmetros mínimos de energia. Cristalino gelatinoso e macio. Priorizar aspiração mecânica pura ou modo I/A."
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
            faco_param = {"Torsional (Ozil)": "40% Pulse", "Faco Longitudinal": "0-5% Linear", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "35 cc/min", "IOP Alvo": "65 mmHg"}
        elif media_v <= 195.0:
            laudo = "G3 - Grau III (Catarata Nuclear Moderada-Avançada)"
            cor = "orange"
            conduta = "Núcleo denso. Obrigatoriedade de técnicas mecânicas de fratura (Faco-Chop ou Quick Chop) para poupar energia ultrassônica total dissipada (CDE)."
            faco_param = {"Torsional (Ozil)": "60% Linear", "Faco Longitudinal": "10% Intelligent Phaco", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "38 cc/min", "IOP Alvo": "70 mmHg Active"}
        else:
            laudo = "G4 - Grau IV (Catarata Nuclear Avançada / Densa Típica)"
            cor = "darkorange"
            conduta = "Cristalino altamente endurecido. Alto risco de perda endotelial e estresse zonular. Injetar viscoelástico dispersivo (Viscoat) repetidas vezes durante o procedimento."
            faco_param = {"Torsional (Ozil)": "80-100% Contínuo", "Faco Longitudinal": "15-20% Intelligent Phaco", "Vácuo Máximo": "450 mmHg", "Fluxo de Aspiração": "40 cc/min", "IOP Alvo": "75 mmHg Active"}

    # 8. Entrega do Laudo na Tela
    st.markdown("---")
    st.markdown("### 📊 Laudo Computacional Automatizado")
    st.subheader(laudo)
    
    # Exibição das métricas analíticas em Tabela Científica
    st.markdown("#### 🔬 Matriz de Parâmetros Ópticos do Miolo")
    dados_metricas = {
        "Métrica Analisada pelo Segmentador": [
            "Brilho Médio do Miolo Profundo (Canal V)", 
            "Saturação de Cor Interna (Canal S)", 
            "Razão Cromática Pura do Miolo (Vermelho / Azul)"
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
    st.markdown("---")
    st.markdown("### ⚙️ Painel Preditivo de Dinâmica de Fluidos (Alcon Centurion)")
    st.caption("Configurações sugeridas no software com base na assinatura de densidade extraída automaticamente do miolo.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Ultrassom Torsional (Ozil)", value=faco_param["Torsional (Ozil)"])
        st.metric(label="Vácuo Máximo", value=faco_param["Vácuo Máximo"])
    with col2:
        st.metric(label="Ultrassom Longitudinal", value=faco_param["Faco Longitudinal"])
        st.metric(label="Fluxo de Aspiração", value=faco_param["Fluxo de Aspiração"])
    with col3:
        st.metric(label="IOP Alvo Estimada", value=faco_param["IOP Alvo"])
        st.metric(label="Estratégia Recomendada", value="Faco-Chop Mecânico" if media_v > 150.0 or cor == "purple" else "Divide & Conquer")
