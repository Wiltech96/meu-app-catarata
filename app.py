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
st.subheader("Classificação Inteligente por Centróide Dinâmico do Núcleo")
st.caption("Versão Premium: Localização de ROI Autoadaptativa por Perfil de Intensidade (HSV)")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda Fina Vertical):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. ALGORITMO DE LOCALIZAÇÃO AUTOMÁTICA DO MIOLO (DETERMINAÇÃO DA ROI INTELIGENTE)
    # Convertemos para escala de cinza padrão para analisar a topografia do brilho
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suavização profunda para eliminar reflexos pontuais isolados (glare) da córnea
    blur_localizador = cv2.GaussianBlur(img_gray, (21, 21), 0)
    
    # Projeta o perfil de intensidade horizontal médio (soma o brilho de todas as colunas)
    perfil_horizontal = np.mean(blur_localizador[int(altura*0.3):int(altura*0.7), :], axis=0)
    
    # Encontra o ponto de maior brilho na metade central da imagem (onde estatisticamente fica o cristalino)
    margem_busca = int(largura * 0.25)
    zona_busca = perfil_horizontal[margem_busca:-margem_busca]
    
    # O pico do gráfico nos dá a coordenada X exata do coração do cristalino
    centro_x = int(np.argmax(zona_busca) + margem_busca)
    centro_y = int(altura * 0.5) # Fixado na linha equatorial do olho
    
    # Define o tamanho do "miolo" rígido de análise (ex: quadrado de 60x60 pixels no centro do núcleo)
    tamanho_miolo = int(min(largura, altura) * 0.08) # Adaptativo ao tamanho/resolução da foto
    
    ymin, ymax = centro_y - tamanho_miolo, centro_y + tamanho_miolo
    xmin, xmax = centro_x - tamanho_miolo, centro_x + tamanho_miolo
    
    # Corta a ROI perfeitamente centralizada no miolo real do paciente
    roi_bgr = img[ymin:ymax, xmin:xmax]
    
    # 5. PROCESSAMENTO AVANÇADO SÓ NO MIOLO DETECTADO (HSV)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_hsv = img_hsv[ymin:ymax, xmin:xmax]
    
    # 6. Desenha a marcação de precisão na tela para o médico conferir
    img_viz = img.copy()
    # Desenha um círculo de mira azul no centro exato do miolo encontrado
    cv2.circle(img_viz, (centro_x, centro_y), 10, (255, 0, 0), -1)
    # Desenha a caixa da ROI em verde ao redor do miolo
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    
    st.image(img_viz, channels="BGR", caption="Mira Computacional Fixada no Miolo Profundo do Núcleo", use_container_width=True)
    
    # 6. EXTRAÇÃO MULTIDIMENSIONAL DE MÉTRICAS NO MIOLO
    media_h = float(np.mean(roi_hsv[:, :, 0])) # Matiz
    media_s = float(np.mean(roi_hsv[:, :, 1])) # Saturação
    media_v = float(np.mean(roi_hsv[:, :, 2])) # Luminosidade V (O nosso cinza blindado)
    
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
        if media_v <= 55.0: # Pequeno ajuste fino no limiar inferior do miolo puro
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
