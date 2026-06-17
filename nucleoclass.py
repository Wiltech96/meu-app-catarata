# --- MOTOR DO NUCLEOCLASS ATUALIZADO (BLINDADO CONTRA REFLEXOS) ---
    h, w = img.shape
    
    # 1. Ajuste de ROI focado estritamente na fenda do cristalino (Descartando a periferia)
    roi = img[int(h*0.3):int(h*0.7), int(w*0.4):int(w*0.65)]
    
    # 2. Filtro Gaussiano adaptativo para suavizar artefatos e ruídos
    roi_suave = cv2.GaussianBlur(roi, (7, 7), 0)
    
    # 3. FILTRO ANTI-REFLEXO: Remove pixels superexpostos (brilho > 240) causados pela córnea ou flashes
    mascara_sem_reflexo = roi_suave[roi_suave < 240]
    
    # Se a imagem estiver muito fora do padrão, evita quebra do sistema
    if len(mascara_sem_reflexo) > 0:
        brilho_real_cristalino = np.mean(mascara_sem_reflexo)
    else:
        brilho_real_cristalino = np.mean(roi_suave)
    
    # 4. Calibração Fina para a Canon Rebel T7 (Ajustada para o seu padrão de ganho óptico)
    min_calibracao, max_calibracao = 25.0, 190.0
    nc_index = max(0.0, min(100.0, ((brilho_real_cristalino - min_calibracao) / (max_calibracao - min_calibracao)) * 100))
    nc_index = round(nc_index, 2)
