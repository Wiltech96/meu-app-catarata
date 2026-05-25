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
st.caption("Desenvolvido para Padronização Digital Baseada no Espaço de Cores HSV")
st.markdown("---")

# 2. Área de Upload da Imagem do Paciente
arquivo = st.file_uploader("Insira a foto da biomicroscopia (Aceita Modo Automático do Celular):", type=["png", "jpg", "jpeg"])

if arquivo is not None:
    # 3. Ler a imagem enviada pelo smartphone
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    altura, largura, _ = img.shape
    
    # 4. RECORTE VERTICAL ANATÔMICO (Trava rigorosamente dentro da fenda e evita vazamento horizontal)
    ymin, ymax = int(altura * 0.35), int(altura * 0.65)  # Mais alto: cobre o núcleo de cima a baixo
    xmin, xmax = int(largura * 0.45), int(largura * 0.55) # Mais estreito: ignora zonas escuras periféricas
    
    # 5. PROCESSAMENTO DIGITAL DE SINAIS (Espaço HSV)
    # Converte a imagem para HSV para neutralizar os ajustes automáticos de brilho da câmera
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    canal_v = img_hsv[:, :, 2]  # Canal V (Value - Luminosidade Pura de 0 a 255)
    
    # Extrai os canais RGB originais estritamente para o cálculo da Razão Cromática
    canal_red = img[:, :, 2]
    canal_blue = img[:, :, 0]
    
    # Extração das médias matemáticas dentro da Região de Interesse (ROI)
    media_v = float(np.mean(canal_v[ymin:ymax, xmin:xmax]))
    media_vermelho = float(np.mean(canal_red[ymin:ymax, xmin:xmax]))
    media_azul = float(np.mean(canal_blue[ymin:ymax, xmin:xmax]))
    
    # Cálculo da razão adaptativa Vermelho/Azul (protegida contra divisão por zero)
    razao_vermelho_azul = media_vermelho / (media_azul + 0.001)
    
    # 6. Desenha o retângulo verde anatômico vertical para conferência do médico
    img_viz = img.copy()
    cv2.rectangle(img_viz, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    st.image(img_viz, channels="BGR", caption="Área de Leitura do Núcleo Cravada na Fenda", use_container_width=True)
    
    # 7. MOTOR DE DECISÃO INTELIGENTE (Régua Baseada em HSV e Razão Cromática)
    
    # Primeiro testa o critério cromático da Catarata Rubra/Brunescente (G6)
    if razao_vermelho_azul > 3.8 and media_v > 80.0:
        laudo = f"G6 - Variante Catarata Rubra / Brunescente Ultra-Densa"
        cor = "purple"
        conduta = "Dureza máxima (rocha). Absorção cromática severa. Exige proteção endotelial máxima (Soft-Shell rígido) e parâmetros de alta energia torsional (Centurion Ozil 100% Contínuo)."
        faco_param = {"Torsional (Ozil)": "100% Contínuo", "Faco Longitudinal": "20-30% em Pulso", "Vácuo Máximo": "450-500 mmHg", "Fluxo de Aspiração": "40-45 cc/min", "IOP Alvo": "80 mmHg"}
    
    # Segundo testa a saturação luminosa cortical da Catarata Branca Total (G5)
    elif media_v >= 210.0:
        laudo = f"G5 - Variante Catarata Branca / Total Intumescente"
        cor = "red"
        conduta = "Opacificação total cortical e estouro de reflexão. Alto risco de hipertensão intralenticular (Sinal da Bandeira Argentina). Realizar descompressão prévia com agulha fina antes da capsulorréxis. Usar Azul de Tripano obrigatório."
        faco_param = {"Torsional (Ozil)": "0% (Usar apenas I/A inicial)", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
    
    # Caso contrário, distribui linearmente pela escala de esclerose filtrada pelo canal V (G0 a G4)
    else:
        if media_v <= 45.0:
            laudo = f"G0 - Cristalino Transparente / Catarata Nuclear Incipiente"
            cor = "green"
            conduta = "Parâmetros mínimos de energia. Cristalino gelatinoso e macio. Priorizar aspiração mecânica pura."
            faco_param = {"Torsional (Ozil)": "0%", "Faco Longitudinal": "0-10% Linear", "Vácuo Máximo": "300 mmHg", "Fluxo de Aspiração": "30 cc/min", "IOP Alvo": "55 mmHg"}
        elif media_v <= 90.0:
            laudo = f"G1 - Grau I (Catarata Nuclear Inicial)"
            cor = "green"
            conduta = "Fragmentação fácil. Baixa densidade nuclear. Parâmetros cirúrgicos conservadores de baixa energia."
            faco_param = {"Torsional (Ozil)": "20% Burst", "Faco Longitudinal": "0% Linear", "Vácuo Máximo": "350 mmHg", "Fluxo de Aspiração": "32 cc/min", "IOP Alvo": "60 mmHg"}
        elif media_v <= 135.0:
            laudo = f"G2 - Grau II (Catarata Nuclear Moderada-Leve)"
            cor = "blue"
            conduta = "Densidade moderada padrão. Fragmentação mecânica fácil. Procedimento convencional estável do serviço."
            faco_param = {"Torsional (Ozil)": "40% Burst/Pulse", "Faco Longitudinal": "0-5% Linear", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "35 cc/min", "IOP Alvo": "65 mmHg"}
        elif media_v <= 180.0:
            laudo = f"G3 - Grau III (Catarata Nuclear Moderada-Avançada)"
            cor = "orange"
            conduta = "Núcleo denso. Obrigatoriedade de técnicas mecânicas de fratura (Faco-Chop ou Quick Chop) para poupar energia ultrassônica total (CDE)."
            faco_param = {"Torsional (Ozil)": "60% Linear", "Faco Longitudinal": "10% Mili-burst", "Vácuo Máximo": "400 mmHg", "Fluxo de Aspiração": "38 cc/min", "IOP Alvo": "70 mmHg"}
        else:
            laudo = f"G4 - Grau IV (Catarata Nuclear Avançada / Densa Típica)"
            cor = "darkorange"
            conduta = "Cristalino altamente endurecido. Alto risco de perda endotelial e estresse zonular. Injetar viscoelástico dispersivo (Viscoat) repetidas vezes durante o procedimento."
            faco_param = {"Torsional (Ozil)": "80-100% Contínuo", "Faco Longitudinal": "15-20% Mili-burst", "Vácuo Máximo": "450 mmHg", "Fluxo de Aspiração": "40 cc/min", "IOP Alvo": "75 mmHg"}

    # 8. Entrega do Laudo e Telemedicina na Tela
    st.markdown("---")
    st.markdown("### 📊 Laudo Computacional")
    st.subheader(laudo)
    
    # Métricas laboratoriais para auditoria do TCC
    st.write(f"🔬 *Métricas Extraídas do Núcleo: Luminosidade V (HSV): {media_v:.1f} | Razão R/A (RGB): {razao_vermelho_azul:.2f}*")
    
    # Exibição do Alerta de Conduta
    if cor in ["purple", "red", "darkorange", "orange"]:
        st.warning(f"⚠️ **Diretriz Cirúrgica:** {conduta}")
    else:
        st.success(f"✅ **Diretriz Cirúrgica:** {conduta}")
        
    # 9. Painel Dinâmico de Parâmetros Alcon Centurion Injetado na Tela
    st.markdown("### ⚙️ Programação Sugerida para Alcon Centurion")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Energia Torsional (Ozil)", faco_param["Torsional (Ozil)"])
        st.metric("Vácuo Máximo", faco_param["Vácuo Máximo"])
        st.metric("Pressão Intraocular (IOP)", faco_param["IOP Alvo"])
    with col2:
        st.metric("Faco Longitudinal", faco_param["Faco Longitudinal"])
        st.metric("Fluxo de Aspiração", faco_param["Fluxo de Aspiração"])
