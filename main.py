import os
import cv2
import math
import joblib
import numpy as np
from collections import deque
from scipy.stats import mode
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import mediapipe as mp
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# =======================
# Configurações
# =======================
train_dir = "data/train"
test_dir = "data/test"

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1,
    refine_landmarks=True,  # 🔹 Melhor detecção
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 🔹 Aumento de dados mais robusto com albumentations
augmentation_pipeline = A.Compose([
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.3),
    A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.3)
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
])

# =======================
# CLASSE: Suavização Temporal Inteligente
# =======================
class SmoothedPredictor:
    """
    Estabiliza previsões usando média móvel ponderada por confiança.
    Implementa filtragem temporal com threshold de confiança.
    """
    def __init__(self, buffer_size=40, confidence_threshold=0.3, use_weighted_avg=True):
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.use_weighted_avg = use_weighted_avg
        
        self.predictions_buffer = deque(maxlen=buffer_size)
        self.confidence_buffer = deque(maxlen=buffer_size)
        self.pred_indices_buffer = deque(maxlen=buffer_size)
        
        self.last_valid_emotion = None
        self.last_valid_confidence = 0
    
    def update(self, pred_idx, confidence):
        """
        Atualiza os buffers com nova predição.
        
        Args:
            pred_idx: Índice codificado da emoção
            confidence: Confiança da predição (0-1)
        """
        self.predictions_buffer.append(pred_idx)
        self.confidence_buffer.append(confidence)
        self.pred_indices_buffer.append(pred_idx)
    
    def get_smoothed_prediction(self):
        if len(self.confidence_buffer) == 0:
            return None, 0.0

        avg_confidence = np.mean(self.confidence_buffer)

        if avg_confidence < self.confidence_threshold:
            return None, avg_confidence

        if self.use_weighted_avg:
            confidence_array = np.array(self.confidence_buffer)
            pred_array = np.array(self.pred_indices_buffer)

            weights = confidence_array / (np.sum(confidence_array) + 1e-8)

            unique_preds = np.unique(pred_array)
            best_pred = unique_preds[0]
            best_weight = 0

            for pred in unique_preds:
                mask = pred_array == pred
                pred_weight = np.sum(weights[mask])
                if pred_weight > best_weight:
                    best_weight = pred_weight
                    best_pred = pred

            # 🔹 Suavização exponencial com último valor válido
            if self.last_valid_emotion is not None:
                alpha = 0.95
                smoothed_pred = int(round(alpha * best_pred + (1 - alpha) * self.last_valid_emotion))
            else:
                smoothed_pred = best_pred
        else:
            result = mode(self.pred_indices_buffer, keepdims=True)
            smoothed_pred = result.mode[0]

        self.last_valid_emotion = smoothed_pred
        self.last_valid_confidence = avg_confidence

        return smoothed_pred, avg_confidence
    
    def reset(self):
        """Reseta todos os buffers."""
        self.predictions_buffer.clear()
        self.confidence_buffer.clear()
        self.pred_indices_buffer.clear()
        self.last_valid_emotion = None
        self.last_valid_confidence = 0

def augment_image(img):
    """Aumento de dados mais sofisticado"""
    augmented = augmentation_pipeline(image=img)
    return augmented['image']

# =======================
# Funções auxiliares melhoradas
# =======================
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle(p1, p2, p3):
    """Calcula o ângulo formado pelos pontos p1-p2-p3 (em graus)."""
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def normalize_face(landmarks):
    """Normalização robusta e estável"""
    # Pontos de referência mais estáveis
    left_eye = np.mean(landmarks[33:133], axis=0)
    right_eye = np.mean(landmarks[362:463], axis=0)
    nose_tip = landmarks[1]
    
    # Centro entre os olhos
    eyes_center = (left_eye + right_eye) / 2
    
    # Centraliza pelo centro dos olhos (mais estável que nariz)
    landmarks = landmarks - eyes_center
    
    # Normaliza pela distância entre os olhos
    eye_dist = distance([0, 0, 0], right_eye - left_eye)
    if eye_dist > 1e-8:
        landmarks = landmarks / eye_dist
    
    # 🔹 Rotação mais precisa
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_rad = np.arctan2(dy, dx)
    
    rot_matrix = np.array([
        [np.cos(-angle_rad), -np.sin(-angle_rad), 0],
        [np.sin(-angle_rad), np.cos(-angle_rad), 0],
        [0, 0, 1]
    ])
    
    landmarks = landmarks @ rot_matrix.T
    return landmarks

def extract_advanced_features(landmarks):
    """
    Extrai features geométricas avançadas com normalização robusta.
    Inclui: distâncias, ângulos, proporções, abertura de olhos/boca, simetrias.
    """
    features = []
    
    # 🔹 Índices expandidos para mais detalhes
    idx = {
        "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        "eyebrows_left": [46, 53, 52, 65, 55, 107, 66, 105, 63, 70, 156],
        "eyebrows_right": [276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 383],
        "nose": [1, 2, 98, 327, 326, 197, 419, 456, 294, 278, 279, 429, 420],
        "mouth_outer": [61, 84, 17, 314, 405, 320, 307, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
        "mouth_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13],
        "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]
    }
    
    # Centros principais
    left_eye_center = np.mean(landmarks[idx["left_eye"]], axis=0)
    right_eye_center = np.mean(landmarks[idx["right_eye"]], axis=0)
    nose_tip = landmarks[1]
    mouth_center = np.mean(landmarks[idx["mouth_outer"]], axis=0)
    left_brow_center = np.mean(landmarks[idx["eyebrows_left"]], axis=0)
    right_brow_center = np.mean(landmarks[idx["eyebrows_right"]], axis=0)
    chin = landmarks[152]
    
    # ======= 1. FEATURES DE DISTÂNCIA EXPANDIDAS =======
    distances = [
        distance(left_eye_center, right_eye_center),
        distance(nose_tip, chin),
        distance(mouth_center, left_eye_center),
        distance(mouth_center, right_eye_center),
        distance(left_brow_center, left_eye_center),
        distance(right_brow_center, right_eye_center),
        distance(left_brow_center, nose_tip),
        distance(right_brow_center, nose_tip),
    ]
    features.extend(distances)
    
    # ======= 2. FEATURES DE ÂNGULO EXPANDIDAS =======
    angles = [
        angle(left_eye_center, nose_tip, right_eye_center),
        angle(left_brow_center, nose_tip, right_brow_center),
        angle(mouth_center, nose_tip, chin),
        angle(left_eye_center, mouth_center, right_eye_center),
    ]
    features.extend(angles)
    
    # ======= 3. ABERTURA DOS OLHOS =======
    # Olho esquerdo
    left_eye_top = np.mean(landmarks[[159, 145]], axis=0)
    left_eye_bottom = landmarks[145]
    left_eye_width = distance(landmarks[33], landmarks[133])
    left_eye_height = distance(left_eye_top, left_eye_bottom)
    
    # Olho direito
    right_eye_top = np.mean(landmarks[[386, 374]], axis=0)
    right_eye_bottom = landmarks[374]
    right_eye_width = distance(landmarks[362], landmarks[263])
    right_eye_height = distance(right_eye_top, right_eye_bottom)
    
    eye_openness = [
        left_eye_height / (left_eye_width + 1e-8),
        right_eye_height / (right_eye_width + 1e-8),
        (left_eye_height + right_eye_height) / (left_eye_width + right_eye_width + 1e-8)
    ]
    features.extend(eye_openness)
    
    # ======= 4. ABERTURA DA BOCA =======
    mouth_width = distance(landmarks[61], landmarks[291])
    mouth_height = distance(landmarks[13], landmarks[14])
    mouth_openness = mouth_height / (mouth_width + 1e-8)
    features.append(mouth_openness)
    
    # ======= 5. SIMETRIA DA BOCA =======
    # Cantos da boca
    mouth_left_corner = landmarks[61]
    mouth_right_corner = landmarks[291]
    mouth_top_left = landmarks[84]
    mouth_top_right = landmarks[314]
    mouth_bottom_left = landmarks[405]
    mouth_bottom_right = landmarks[320]
    
    # Diferenças entre cantos (assimetria)
    mouth_corner_symmetry = abs(
        distance(mouth_top_left, mouth_left_corner) - 
        distance(mouth_top_right, mouth_right_corner)
    )
    features.append(mouth_corner_symmetry)
    
    # Altura dos lados da boca
    mouth_left_height = distance(mouth_top_left, mouth_bottom_left)
    mouth_right_height = distance(mouth_top_right, mouth_bottom_right)
    mouth_height_asymmetry = abs(mouth_left_height - mouth_right_height)
    features.append(mouth_height_asymmetry)
    
    # ======= 6. ELEVAÇÃO DA SOBRANCELHA =======
    brow_raise_left = distance(left_brow_center, left_eye_center)
    brow_raise_right = distance(right_brow_center, right_eye_center)
    features.extend([brow_raise_left, brow_raise_right])
    
    # ======= 7. ASSIMETRIAS FACIAIS =======
    vertical_symmetry = abs(distance(left_eye_center, nose_tip) - distance(right_eye_center, nose_tip))
    horizontal_symmetry = abs(distance(left_brow_center, mouth_center) - distance(right_brow_center, mouth_center))
    features.extend([vertical_symmetry, horizontal_symmetry])
    
    # ======= 8. CURVATURA DOS LÁBIOS =======
    lip_curve = distance(landmarks[13], (landmarks[61] + landmarks[291]) / 2)
    features.append(lip_curve)
    
    # ======= 9. FEATURES DE ÁREA RELATIVA =======
    face_width = distance(landmarks[234], landmarks[454])
    face_height = distance(landmarks[10], landmarks[152])
    
    area_features = [
        mouth_width / (face_width + 1e-8),
        mouth_height / (face_height + 1e-8),
        distance(left_eye_center, right_eye_center) / (face_width + 1e-8)
    ]
    features.extend(area_features)
    
    # ======= 10. POSIÇÃO DOS OLHOS EM RELAÇÃO AO NARIZ =======
    left_eye_rel_x = (left_eye_center[0] - nose_tip[0]) / (distance(left_eye_center, right_eye_center) + 1e-8)
    left_eye_rel_y = (left_eye_center[1] - nose_tip[1]) / (distance(left_eye_center, right_eye_center) + 1e-8)
    right_eye_rel_x = (right_eye_center[0] - nose_tip[0]) / (distance(left_eye_center, right_eye_center) + 1e-8)
    right_eye_rel_y = (right_eye_center[1] - nose_tip[1]) / (distance(left_eye_center, right_eye_center) + 1e-8)
    
    features.extend([left_eye_rel_x, left_eye_rel_y, right_eye_rel_x, right_eye_rel_y])
    
    return np.array(features, dtype=np.float32)

def extract_features(image):
    """Função principal de extração de features"""
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
    landmarks = normalize_face(landmarks)
    
    features = extract_advanced_features(landmarks)
    return features

# =======================
# Carregamento de dados com balanceamento
# =======================
def load_data(data_dir, augment=True):
    X, y = [], []
    emotion_counts = {emotion: 0 for emotion in emotions}
    
    for emotion in emotions:
        path = os.path.join(data_dir, emotion)
        if not os.path.exists(path):
            print(f"⚠️  Pasta não encontrada: {path}")
            continue
            
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # 🔹 Redimensionamento mantendo proporção
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Feature original
            features = extract_features(img)
            if features is not None:
                X.append(features)
                y.append(emotion)
                emotion_counts[emotion] += 1
            
            # Aumento de dados
            if augment:
                for _ in range(3):  # 🔹 Mais augmentações
                    try:
                        img_aug = augment_image(img)
                        features_aug = extract_features(img_aug)
                        if features_aug is not None:
                            X.append(features_aug)
                            y.append(emotion)
                            emotion_counts[emotion] += 1
                    except Exception as e:
                        continue
    
    print("📊 Distribuição dos dados de TREINO:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} amostras")
        
    return np.array(X), np.array(y)

# =======================
# Modelos otimizados
# =======================
def create_mlp_model(n_features):
    """MLP otimizado com regularização"""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            learning_rate='adaptive',
            validation_fraction=0.2,
            alpha=1e-4,
            batch_size=16,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=50,
            random_state=42,
            verbose=0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
    )

def create_random_forest_model(class_weights=None):
    """RandomForest mais robusto"""
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    )

def create_gradient_boosting_model(class_weights=None):
    """GradientBoosting com regularização"""
    return make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42,
            verbose=0
        )
    )

def evaluate_model(model, X_val, y_val, le, model_name="Model"):
    """Avalia modelo com múltiplas métricas"""
    y_pred = model.predict(X_val)
    
    acc = (y_pred == y_val).mean()
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"📊 MÉTRICAS DO MODELO: {model_name}")
    print(f"{'='*60}")
    print(f"✅ Acurácia:  {acc*100:.2f}%")
    print(f"✅ Precisão: {precision*100:.2f}%")
    print(f"✅ Recall:   {recall*100:.2f}%")
    print(f"✅ F1-Score: {f1*100:.2f}%")
    print(f"{'='*60}")
    
    print("\n📋 Relatório de classificação detalhado:")
    print(classification_report(y_val, y_pred, target_names=le.classes_, digits=3))
    
    print("\n🎯 Matriz de confusão (normalizada):")
    cm = confusion_matrix(y_val, y_pred, labels=le.transform(le.classes_))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    print(pd.DataFrame(cm_normalized, index=le.classes_, columns=le.classes_).round(3))
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred': y_pred
    }

def train_model(model_type='all'):
    """
    Treina e avalia múltiplos modelos com validação cruzada.
    
    Args:
        model_type: 'mlp', 'rf', 'gb', ou 'all'
    """
    print("🔹 Carregando dados de treino...")
    X_train, y_train = load_data(train_dir, augment=True)
    
    if len(X_train) == 0:
        print("❌ Nenhum dado carregado! Verifique os caminhos e imagens.")
        return
        
    print(f"✅ Dados carregados: {X_train.shape}, {y_train.shape}")
    
    # 🔹 Encoding de labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    # 🔹 Balanceamento com SMOTE
    print("\n🔹 Aplicando SMOTE para balanceamento de classes...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_enc)
        print(f"✅ Dados balanceados: {X_train_balanced.shape}")
    except:
        print("⚠️  SMOTE falhou, usando dados originais")
        X_train_balanced, y_train_balanced = X_train, y_train_enc
    
    # 🔹 Split para validação
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_balanced, y_train_balanced, test_size=0.2, 
        random_state=42, stratify=y_train_balanced
    )
    
    print(f"📚 Treino: {X_train_split.shape}, Validação: {X_val_split.shape}")
    
    # Calcular class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train_split),
        y=y_train_split
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    models_to_train = []
    if model_type in ['mlp', 'all']:
        models_to_train.append(('MLP', create_mlp_model(X_train.shape[1]), None))
    if model_type in ['rf', 'all']:
        models_to_train.append(('RandomForest', create_random_forest_model(class_weight_dict), class_weight_dict))
    if model_type in ['gb', 'all']:
        models_to_train.append(('GradientBoosting', create_gradient_boosting_model(class_weight_dict), class_weight_dict))
    
    best_model = None
    best_f1 = 0
    best_model_name = ""
    results_summary = []
    
    for model_name, model, _ in models_to_train:
        print(f"\n🚀 Treinando {model_name}...")
        model.fit(X_train_split, y_train_split)
        
        metrics = evaluate_model(model, X_val_split, y_val_split, le, model_name)
        results_summary.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1']
        })
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model
            best_model_name = model_name
    
    # 🔹 Validação cruzada do melhor modelo
    print(f"\n{'='*60}")
    print(f"🏆 MELHOR MODELO: {best_model_name} (F1-Score: {best_f1*100:.2f}%)")
    print(f"{'='*60}")
    
    print(f"\n� Aplicando validação cruzada (5-fold) no {best_model_name}...")
    cv_scores = cross_val_score(
        best_model, X_train_balanced, y_train_balanced, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted'
    )
    print(f"✅ CV Scores: {cv_scores}")
    print(f"✅ CV Mean F1-Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    # 🔹 Salvar melhor modelo
    joblib.dump(best_model, f"fer_{best_model_name.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print(f"\n💾 Melhor modelo ({best_model_name}) salvo!")
    
    # Salvar resumo
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv("model_comparison.csv", index=False)
    print("💾 Comparação de modelos salva em model_comparison.csv")
    
    return best_model, le

# =======================
# Funções de teste e tempo real melhoradas
# =======================
def realtime(model_file=None):
    """
    Executa detecção em tempo real com suavização temporal inteligente.
    
    Args:
        model_file: Path do modelo. Se None, usa o melhor modelo disponível.
    """
    print("📦 Carregando modelo...")
    try:
        if model_file is None:
            # Tenta carregar na ordem de preferência
            for model_name in ['fer_gradientboosting_model.pkl', 'fer_randomforest_model.pkl', 'fer_mlp_model.pkl']:
                if os.path.exists(model_name):
                    model_file = model_name
                    break
        
        if model_file is None:
            print("❌ Nenhum modelo encontrado. Execute --train primeiro.")
            return
        
        model = joblib.load(model_file)
        le = joblib.load("label_encoder.pkl")
        print(f"✅ Modelo carregado: {model_file}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return

    cap = cv2.VideoCapture(0)
    
    # 🔹 Suavização inteligente
    predictor = SmoothedPredictor(
        buffer_size=40,
        confidence_threshold=0.15,
        use_weighted_avg=True
    )
    
    frame_count = 0
    fps_counter = 0
    import time
    start_time = time.time()

    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as face_mesh_rt:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            fps_counter += 1
            
            # Calcula FPS
            elapsed = time.time() - start_time
            if elapsed > 1:
                fps = fps_counter / elapsed
                fps_counter = 0
                start_time = time.time()
            else:
                fps = 0
            
            # Extrai features
            features = extract_features(frame)
            
            if features is not None:
                # Predição
                pred_idx = model.predict([features])[0]
                proba = model.predict_proba([features])[0]
                confidence = np.max(proba)
                
                # Atualiza predictor
                predictor.update(pred_idx, confidence)
                
                # Obtém predição suavizada
                smoothed_idx, avg_confidence = predictor.get_smoothed_prediction()
                
                # Exibe resultado
                if smoothed_idx is not None:
                    emotion = le.inverse_transform([smoothed_idx])[0]
                    color = (0, 255, 0) if avg_confidence > 0.6 else (0, 165, 255) if avg_confidence > 0.5 else (0, 0, 255)
                    
                    # Texto com emoção e confiança
                    text = f"{emotion.upper()} ({avg_confidence:.1%})"
                    cv2.putText(frame, text, (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                else:
                    # Confiança baixa
                    text = f"INDEFINIDO ({avg_confidence:.1%})"
                    cv2.putText(frame, text, (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
                
                # Info adicional
                info_text = f"Raw conf: {confidence:.1%} | Buffer: {len(predictor.confidence_buffer)}"
                cv2.putText(frame, info_text, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "Nenhum rosto detectado", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # FPS
            if fps > 0:
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("FER - Reconhecimento de Emoções (Press ESC to exit)", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Execução concluída! Total de frames processados: {frame_count}")

def test_images(test_dir, model_file=None):
    """
    Testa modelo em imagens da pasta de teste.
    
    Args:
        test_dir: Diretório de teste
        model_file: Path do modelo. Se None, usa o melhor disponível.
    """
    print("📦 Carregando modelo...")
    try:
        if model_file is None:
            for model_name in ['fer_gradientboosting_model.pkl', 'fer_randomforest_model.pkl', 'fer_mlp_model.pkl']:
                if os.path.exists(model_name):
                    model_file = model_name
                    break
        
        if model_file is None:
            print("❌ Nenhum modelo encontrado. Execute --train primeiro.")
            return
        
        model = joblib.load(model_file)
        le = joblib.load("label_encoder.pkl")
        print(f"✅ Modelo carregado: {model_file}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return

    results = []
    all_true, all_pred = [], []
    confidence_per_emotion = {emotion: [] for emotion in emotions}

    print(f"\n🔹 Testando imagens de {test_dir}...")
    
    for emotion in emotions:
        emotion_path = os.path.join(test_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
        
        print(f"  Processando {emotion}...", end=" ")
        emotion_count = 0
        
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            features = extract_features(img)
            if features is None:
                results.append({
                    "image": img_name, 
                    "true_emotion": emotion, 
                    "predicted_emotion": "no_face",
                    "confidence": 0.0,
                    "correct": False
                })
                continue

            pred_idx = model.predict([features])[0]
            proba = model.predict_proba([features])[0]
            confidence = np.max(proba)
            
            predicted_emotion = le.inverse_transform([pred_idx])[0]
            is_correct = (emotion == predicted_emotion)
            
            results.append({
                "image": img_name, 
                "true_emotion": emotion, 
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
                "correct": is_correct
            })
            
            if emotion == predicted_emotion:
                all_true.append(emotion)
                all_pred.append(predicted_emotion)
                confidence_per_emotion[emotion].append(confidence)
            else:
                all_true.append(emotion)
                all_pred.append(predicted_emotion)
            
            emotion_count += 1
        
        print(f"({emotion_count} imagens)")

    # Salva resultados detalhados
    df = pd.DataFrame(results)
    df.to_csv("test_results_improved.csv", index=False)
    print(f"\n✅ Predições salvas em test_results_improved.csv")

    # Análise detalhada
    df_valid = df[df["predicted_emotion"] != "no_face"]
    
    if len(df_valid) > 0:
        overall_accuracy = (df_valid["true_emotion"] == df_valid["predicted_emotion"]).mean()
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTADOS DO TESTE")
        print(f"{'='*60}")
        print(f"✅ Total de imagens: {len(df_valid)}")
        print(f"✅ Acurácia geral: {overall_accuracy*100:.2f}%")
        print(f"✅ Confiança média: {df_valid['confidence'].mean()*100:.2f}%")
        print(f"{'='*60}")
        
        # Matriz de confusão
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_true, all_pred, labels=emotions)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        print("\n🎯 Matriz de confusão (normalizada):")
        cm_df = pd.DataFrame(cm_normalized, index=emotions, columns=emotions)
        print(cm_df.round(3).to_string())
        
        # Acurácia por emoção
        print(f"\n📊 Acurácia por emoção:")
        per_emotion_acc = {}
        for emotion in emotions:
            emotion_mask = df_valid["true_emotion"] == emotion
            if emotion_mask.sum() > 0:
                acc = (df_valid.loc[emotion_mask, "true_emotion"] == 
                       df_valid.loc[emotion_mask, "predicted_emotion"]).mean()
                per_emotion_acc[emotion] = acc
                print(f"  {emotion}: {acc*100:.2f}%")
        
        # Confiança média por emoção
        print(f"\n📊 Confiança média por emoção:")
        for emotion in emotions:
            emotion_mask = df_valid["true_emotion"] == emotion
            if emotion_mask.sum() > 0:
                conf = df_valid.loc[emotion_mask, "confidence"].mean()
                print(f"  {emotion}: {conf*100:.2f}%")
        
        # Exemplos de acertos e erros
        print(f"\n✅ Primeiras 10 ACERTOS:")
        correct_df = df_valid[df_valid["correct"] == True].head(10)
        print(correct_df[["image", "true_emotion", "confidence"]].to_string(index=False))
        
        print(f"\n❌ Primeiras 10 ERROS:")
        wrong_df = df_valid[df_valid["correct"] == False].head(10)
        print(wrong_df[["image", "true_emotion", "predicted_emotion", "confidence"]].to_string(index=False))
    else:
        print("❌ Nenhuma predição válida encontrada!")

# =======================
# Execução principal
# =======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sistema de Reconhecimento de Emoções Faciais com MediaPipe"
    )
    parser.add_argument("--train", 
                       choices=['mlp', 'rf', 'gb', 'all'],
                       nargs='?', const='all',
                       help="Treina modelo(s): mlp, rf, gb, ou all")
    parser.add_argument("--realtime", action="store_true", 
                       help="Executa detecção em tempo real (webcam)")
    parser.add_argument("--test", action="store_true",
                       help="Testa imagens da pasta de teste")
    parser.add_argument("--model", type=str,
                       help="Especifica arquivo do modelo a usar (ex: fer_randomforest_model.pkl)")
    
    args = parser.parse_args()

    if args.train:
        train_model(model_type=args.train)
    elif args.realtime:
        realtime(model_file=args.model)
    elif args.test:
        test_images(test_dir, model_file=args.model)
    else:
        print("="*70)
        print("🎭 SISTEMA DE RECONHECIMENTO DE EMOÇÕES FACIAIS")
        print("="*70)
        print("\n📖 USO:")
        print("  python main.py --train [mlp|rf|gb|all]  # Treina um ou múltiplos modelos")
        print("  python main.py --realtime               # Webcam em tempo real")
        print("  python main.py --test                   # Testa imagens")
        print("  python main.py --realtime --model FILE  # Usa modelo específico")
        print("\n✨ MODELOS DISPONÍVEIS:")
        print("  - mlp: MLPClassifier (rede neural)")
        print("  - rf:  RandomForest (ensemble)")
        print("  - gb:  GradientBoosting (ensemble)")
        print("  - all: Treina e compara todos os 3 modelos")
        print("="*70)
        print("\n💡 EXEMPLOS:")
        print("  # Treinar todos os modelos:")
        print("  python main.py --train all")
        print("\n  # Usar modelo RandomForest em tempo real:")
        print("  python main.py --realtime --model fer_randomforest_model.pkl")
        print("\n  # Testar com GradientBoosting:")
        print("  python main.py --test --model fer_gradientboosting_model.pkl")
        print("="*70)