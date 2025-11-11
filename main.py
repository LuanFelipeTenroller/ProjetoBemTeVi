import os
import cv2
import math
import joblib
import numpy as np
from collections import deque
from scipy.stats import mode
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mediapipe as mp
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
])

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
    """Normalização mais robusta"""
    # Pontos de referência mais estáveis
    left_eye = np.mean(landmarks[33:133], axis=0)
    right_eye = np.mean(landmarks[362:463], axis=0)
    nose_tip = landmarks[1]
    
    # Centro entre os olhos
    eyes_center = (left_eye + right_eye) / 2
    
    # Centraliza pelo centro dos olhos (mais estável que nariz)
    landmarks -= eyes_center
    
    # Normaliza pela distância entre os olhos
    eye_dist = distance(left_eye, right_eye)
    if eye_dist > 0:
        landmarks /= eye_dist
    
    # 🔹 Rotação mais precisa
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle_rad = np.arctan2(dy, dx)
    
    rot_matrix = np.array([
        [np.cos(-angle_rad), -np.sin(-angle_rad), 0],
        [np.sin(-angle_rad), np.cos(-angle_rad), 0],
        [0, 0, 1]
    ])
    
    landmarks = landmarks @ rot_matrix.T
    return landmarks

def extract_advanced_features(landmarks):
    """Extrai features geométricas avançadas e relações espaciais"""
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
    
    # 1. FEATURES DE DISTÂNCIA EXPANDIDAS
    left_eye_center = np.mean(landmarks[idx["left_eye"]], axis=0)
    right_eye_center = np.mean(landmarks[idx["right_eye"]], axis=0)
    nose_tip = landmarks[1]
    mouth_center = np.mean(landmarks[idx["mouth_outer"]], axis=0)
    left_brow_center = np.mean(landmarks[idx["eyebrows_left"]], axis=0)
    right_brow_center = np.mean(landmarks[idx["eyebrows_right"]], axis=0)
    chin = landmarks[152]
    
    # Distâncias principais
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
    
    # 2. FEATURES DE ÂNGULO EXPANDIDAS
    angles = [
        angle(left_eye_center, nose_tip, right_eye_center),  # Ângulo entre olhos-nariz
        angle(left_brow_center, nose_tip, right_brow_center),  # Ângulo entre sobrancelhas
        angle(mouth_center, nose_tip, chin),  # Ângulo vertical do rosto
        angle(left_eye_center, mouth_center, right_eye_center),  # Ângulo da boca em relação aos olhos
    ]
    features.extend(angles)
    
    # 3. FEATURES DE PROPORÇÃO E FORMA
    # Altura/largura dos olhos
    left_eye_width = distance(landmarks[33], landmarks[133])
    left_eye_height = distance(landmarks[159], landmarks[145])
    right_eye_width = distance(landmarks[362], landmarks[263])
    right_eye_height = distance(landmarks[386], landmarks[374])
    
    eye_ratios = [
        left_eye_height / (left_eye_width + 1e-8),
        right_eye_height / (right_eye_width + 1e-8),
        (left_eye_height + right_eye_height) / (left_eye_width + right_eye_width + 1e-8)
    ]
    features.extend(eye_ratios)
    
    # 4. FEATURES DE EXPRESSÃO ESPECÍFICAS
    # Abertura da boca
    mouth_width = distance(landmarks[61], landmarks[291])
    mouth_height = distance(landmarks[13], landmarks[14])
    mouth_openness = mouth_height / (mouth_width + 1e-8)
    features.append(mouth_openness)
    
    # Sobrancelhas
    brow_raise_left = distance(left_brow_center, left_eye_center)
    brow_raise_right = distance(right_brow_center, right_eye_center)
    features.extend([brow_raise_left, brow_raise_right])
    
    # 5. ASSIMETRIAS FACIAIS
    vertical_symmetry = abs(distance(left_eye_center, nose_tip) - distance(right_eye_center, nose_tip))
    horizontal_symmetry = abs(distance(left_brow_center, mouth_center) - distance(right_brow_center, mouth_center))
    features.extend([vertical_symmetry, horizontal_symmetry])
    
    # 6. FEATURES DE CURVATURA E ORIENTAÇÃO
    # Curvatura dos lábios
    lip_curve = distance(landmarks[13], (landmarks[61] + landmarks[291]) / 2)
    features.append(lip_curve)
    
    # 7. FEATURES DE ÁREA RELATIVA (aproximada)
    face_width = distance(landmarks[234], landmarks[454])  # Bochechas
    face_height = distance(landmarks[10], landmarks[152])  # Testa-queixo
    
    area_features = [
        mouth_width / (face_width + 1e-8),
        mouth_height / (face_height + 1e-8),
        distance(left_eye_center, right_eye_center) / (face_width + 1e-8)
    ]
    features.extend(area_features)
    
    return np.array(features)

def extract_features(image):
    """Função principal de extração de features"""
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
    landmarks = normalize_face(landmarks)
    
    return extract_advanced_features(landmarks)

# =======================
# Carregamento de dados melhorado
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
    
    print("📊 Distribuição dos dados:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} amostras")
        
    return np.array(X), np.array(y)

# =======================
# Modelo melhorado
# =======================
def create_improved_model():
    """Cria modelo MLP otimizado"""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(1024, 512, 256, 128),  # 🔹 Arquitetura mais profunda
            activation='relu',
            solver='adam',
            learning_rate_init=0.0005,  # 🔹 Learning rate ajustado
            learning_rate='adaptive',
            validation_fraction=0.2,
            alpha=1e-5,  # 🔹 Regularização mais suave
            batch_size=32,  # 🔹 Batch menor para melhor generalização
            max_iter=500,   # 🔹 Mais épocas
            early_stopping=True,
            n_iter_no_change=30,
            random_state=42,
            verbose=True,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
    )

def train_model():
    print("🔹 Carregando dados de treino...")
    X_train, y_train = load_data(train_dir, augment=True)
    
    if len(X_train) == 0:
        print("❌ Nenhum dado carregado! Verifique os caminhos e imagens.")
        return
        
    print(f"✅ Dados carregados: {X_train.shape}, {y_train.shape}")

    # 🔹 Balanceamento de classes
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    # 🔹 Split para validação
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train_enc, test_size=0.2, random_state=42, stratify=y_train_enc
    )
    
    print(f"📚 Treino: {X_train_split.shape}, Validação: {X_val_split.shape}")
    
    model = create_improved_model()

    print("🚀 Iniciando treinamento...")
    model.fit(X_train_split, y_train_split)
    print("✅ Treinamento concluído!")
    
    # 🔹 Avaliação no conjunto de validação
    val_score = model.score(X_val_split, y_val_split)
    print(f"📊 Acurácia na validação: {val_score*100:.2f}%")
    
    # Predições detalhadas
    y_pred = model.predict(X_val_split)
    print("\n📋 Relatório de classificação:")
    print(classification_report(y_val_split, y_pred, target_names=le.classes_))
    
    # Salvar modelo
    joblib.dump(model, "fer_mlp_improved_model.pkl")
    joblib.dump(le, "label_encoder_improved.pkl")
    print("💾 Modelo e encoder salvos!")

# =======================
# Funções de teste e tempo real (mantidas similares)
# =======================
def realtime():
    print("📦 Carregando modelo melhorado...")
    try:
        model = joblib.load("fer_mlp_improved_model.pkl")
        le = joblib.load("label_encoder_improved.pkl")
    except:
        print("❌ Modelo não encontrado. Execute --train primeiro.")
        return

    cap = cv2.VideoCapture(0)
    buffer_size = 15  # 🔹 Buffer maior para mais suavização
    predictions_buffer = deque(maxlen=buffer_size)
    confidence_buffer = deque(maxlen=buffer_size)

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

            features = extract_features(frame)
            if features is not None:
                pred = model.predict([features])[0]
                proba = model.predict_proba([features])[0]
                confidence = np.max(proba)
                
                predictions_buffer.append(pred)
                confidence_buffer.append(confidence)
                
                if len(predictions_buffer) >= buffer_size // 2:
                    result = mode(predictions_buffer, keepdims=True)
                    smoothed_pred = result.mode[0]
                    avg_confidence = np.mean(confidence_buffer)
                    
                    emotion = le.inverse_transform([smoothed_pred])[0]
                    
                    # 🔹 Mostrar confiança
                    text = f"{emotion} ({avg_confidence:.2f})"
                    color = (0, 255, 0) if avg_confidence > 0.6 else (0, 165, 255)
                    
                    cv2.putText(frame, text, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("FER - Improved Model", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

def test_images(test_dir):
    print("📦 Carregando modelo melhorado...")
    try:
        model = joblib.load("fer_mlp_improved_model.pkl")
        le = joblib.load("label_encoder_improved.pkl")
    except:
        print("❌ Modelo não encontrado. Execute --train primeiro.")
        return

    results = []
    all_true, all_pred = [], []

    for emotion in emotions:
        emotion_path = os.path.join(test_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
            
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
                    "predicted_emotion": "no_face"
                })
                continue

            pred = model.predict([features])[0]
            proba = model.predict_proba([features])[0]
            confidence = np.max(proba)
            
            predicted_emotion = le.inverse_transform([pred])[0]
            
            results.append({
                "image": img_name, 
                "true_emotion": emotion, 
                "predicted_emotion": predicted_emotion,
                "confidence": confidence
            })
            
            all_true.append(emotion)
            all_pred.append(predicted_emotion)

    # Salva resultados
    df = pd.DataFrame(results)
    df.to_csv("test_results_improved.csv", index=False)
    print("✅ Predições salvas em test_results_improved.csv")

    # Calcula acurácia
    df_valid = df[df["predicted_emotion"] != "no_face"]
    if len(df_valid) > 0:
        accuracy = (df_valid["true_emotion"] == df_valid["predicted_emotion"]).mean()
        print(f"📊 Acurácia: {accuracy*100:.2f}%")
        
        # 🔹 Matriz de confusão
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_true, all_pred, labels=emotions)
        print("\n🎯 Matriz de confusão:")
        print(pd.DataFrame(cm, index=emotions, columns=emotions))
    else:
        print("❌ Nenhuma predição válida encontrada!")

    print(df_valid.head(15))

# =======================
# Execução principal
# =======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Treina o modelo")
    parser.add_argument("--realtime", action="store_true", help="Executa webcam")
    parser.add_argument("--test", action="store_true", help="Testa imagens da pasta")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.realtime:
        realtime()
    elif args.test: 
        test_images(test_dir)
    else:
        print("Use --train para treinar, --realtime para rodar a webcam ou --test para testar imagens.")