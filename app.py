import streamlit as st
import cv2
import numpy as np

# 1. Configuração da Identidade Visual Médica do Software
st.set_page_config(
    page_title="NucleoClass Auto", 
    layout="centered", 
    page_icon="👁️"
)
st.title("👁️ NucleoClass - Automação por Varredura de Intensidade")
st.subheader("Classificação Automatizada com Filtro Anti-Cápsula")
st.caption("Versão Final: Deslocamento Anatômico Calculado para Isolamento do Núcleo")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Fenda Fina Vertical):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. FILTRO DE CÓRNEA: Limita a busca ao centro da pupila (35% a 65%)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_inicio_busca = int(largura * 0.35)
    x_fim_busca = int(largura * 0.65)
    
    roi_busca = img_gray[int(altura*0.4):int(altura*0.6), x_inicio_busca:x_fim_busca]
    
    # Encontra a coluna mais brilhante (A cápsula anterior hiper-reflexiva)
    perfil_luminoso = np.mean(roi_busca, axis=0)
    coluna_pico_relativa = int(np.argmax(perfil_luminoso))
    coluna_capsula_x = x_inicio_busca + coluna_pico_relativa
    
    # 5. PULO DO GATO ANATÔMICO: Deslocamento horizontal para trás da cápsula anterior
    # Movemos a leitura cerca de 6% da largura total da imagem para dentro do cristalino (ajustado para fenda clássica)
    # Se o feixe luminoso vem da direita para a esquerda, o núcleo fica à esquerda da cápsula (-).
    # Caso o seu serviço use o feixe vindo do outro lado, basta mudar o sinal de (-) para (+) abaixo.
    coluna_nucleo_x = coluna_capsula_x - int(largura * 0.06)
    
    # Garantia de limites da imagem
    if coluna_nucleo_x < int(largura*0.2) or coluna_nucleo_x > int(largura*0.8):
        coluna_nucleo_x = int(largura * 0.5)
        
    # 6. DEFINE A ROI AUTOMÁTICA CENTRADA NO NÚCLEO PROFUNDO (Filete estreito vertical)
    ymin, ymax = int(altura * 0.42), int(altura * 0.58)  # Mais concentrado na vertical para fugir das bordas corticais
    xmin, xmax = max(0, coluna_nucleo_x - int(largura * 0.03)), min(largura, coluna_nucleo_x + int(largura * 0.03))
    
    # 7. PROCESSAMENTO DIGITAL DE SINAIS (Espaço HSV e RGB)
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
    
    # 8. EXIBIÇÃO VISUAL DO ENQUADRAMENTO DA IA
    img_viz = img.copy()
    # Desenha uma linha fina amarela onde o software achou a Cápsula (Apenas para checagem do médico)
    cv2.line(img_viz, (coluna_capsula_x, ymin), (coluna_capsula_x, ymax), (0, 255, 255), 2)
    # Desenha o retângulo verde oficial cravado no Núcleo profundo
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Linha Amarela: Cápsula Identificada | Retângulo Verde: Núcleo Profundo Isolado", use_container_width=True)
    
    # 9. MOTOR DE DECISÃO AUTOMÁTICO COM RÉGUA CALIBRADA PELO SMARTPHONE
    if media_s < 30.0 and media_v > 40.0:
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
