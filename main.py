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
import mediapipe as mp
import pandas as pd

# =======================
# Configurações
# =======================
train_dir = "data/train"
test_dir = "data/test"

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

# =======================
# Funções auxiliares
# =======================
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle(p1, p2, p3):
    """Calcula o ângulo formado pelos pontos p1-p2-p3 (em graus)."""
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def normalize_face(landmarks):
    """Centraliza e normaliza a face em relação aos olhos e nariz."""
    left_eye = np.mean(landmarks[33:133], axis=0)
    right_eye = np.mean(landmarks[362:463], axis=0)
    nose = landmarks[1]

    landmarks -= nose
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist > 0:
        landmarks /= eye_dist

    return landmarks

def extract_features(image):
    """Extrai features geométricas avançadas da face."""
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
    landmarks = normalize_face(landmarks)

    idx = {
        "left_eye": [33, 133, 159, 145],
        "right_eye": [362, 263, 386, 374],
        "nose": [1, 2, 98],
        "mouth": [61, 291, 78, 308],
        "chin": [152, 199],
        "brow_left": [105, 55, 70],
        "brow_right": [334, 285, 300],
        "cheek_left": [234, 93],
        "cheek_right": [454, 323]
    }

    features = []

    # Centros e pontos principais
    left_eye_center = np.mean(landmarks[idx["left_eye"]], axis=0)
    right_eye_center = np.mean(landmarks[idx["right_eye"]], axis=0)
    nose_center = np.mean(landmarks[idx["nose"]], axis=0)
    mouth_left, mouth_right = landmarks[idx["mouth"][0]], landmarks[idx["mouth"][1]]
    mouth_top, mouth_bottom = landmarks[idx["mouth"][2]], landmarks[idx["mouth"][3]]
    brow_left_mid = np.mean(landmarks[idx["brow_left"]], axis=0)
    brow_right_mid = np.mean(landmarks[idx["brow_right"]], axis=0)
    chin_center = np.mean(landmarks[idx["chin"]], axis=0)

    # ===========================
    # 📏 Distâncias básicas
    # ===========================
    features += [
        distance(left_eye_center, right_eye_center),
        distance(nose_center, chin_center),
        distance(mouth_left, mouth_right),
        distance(mouth_top, mouth_bottom),
        distance(left_eye_center, mouth_left),
        distance(right_eye_center, mouth_right),
        distance(left_eye_center, mouth_top),
        distance(right_eye_center, mouth_top),
        distance(brow_left_mid, left_eye_center),
        distance(brow_right_mid, right_eye_center),
        distance(brow_left_mid, mouth_left),
        distance(brow_right_mid, mouth_right)
    ]

    # ===========================
    # 🧭 Ângulos
    # ===========================
    features += [
        angle(left_eye_center, nose_center, right_eye_center),
        angle(mouth_left, nose_center, mouth_right),
        angle(brow_left_mid, nose_center, brow_right_mid),
        angle(nose_center, mouth_top, mouth_bottom),
        angle(brow_left_mid, left_eye_center, mouth_left),
        angle(brow_right_mid, right_eye_center, mouth_right)
    ]

    # ===========================
    # 📊 Razões e simetrias
    # ===========================
    mouth_ratio = distance(mouth_top, mouth_bottom) / (distance(mouth_left, mouth_right) + 1e-6)
    eye_ratio = (
        distance(landmarks[idx["left_eye"][1]], landmarks[idx["left_eye"][3]]) +
        distance(landmarks[idx["right_eye"][1]], landmarks[idx["right_eye"][3]])
    ) / (2 * distance(left_eye_center, right_eye_center) + 1e-6)

    brow_eye_ratio = (
        (distance(brow_left_mid, left_eye_center) + distance(brow_right_mid, right_eye_center)) /
        (2 * distance(left_eye_center, right_eye_center) + 1e-6)
    )

    symmetry = abs(distance(left_eye_center, nose_center) - distance(right_eye_center, nose_center))
    mouth_symmetry = abs(distance(mouth_left, nose_center) - distance(mouth_right, nose_center))

    features += [mouth_ratio, eye_ratio, brow_eye_ratio, symmetry, mouth_symmetry]


    left_eye_height = distance(landmarks[idx["left_eye"][2]], landmarks[idx["left_eye"][3]])
    right_eye_height = distance(landmarks[idx["right_eye"][2]], landmarks[idx["right_eye"][3]])
    features += [left_eye_height, right_eye_height, (left_eye_height + right_eye_height) / 2]

    return np.array(features)


# =======================
# Carregamento dos dados
# =======================
def load_data(data_dir):
    X, y = [], []
    for emotion in emotions:
        path = os.path.join(data_dir, emotion)
        if not os.path.exists(path):
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))
            features = extract_features(img)
            if features is not None:
                X.append(features)
                y.append(emotion)
    return np.array(X), np.array(y)

# =======================
# Treinamento
# =======================
def train_model():
    print("🔹 Carregando dados de treino...")
    X_train, y_train = load_data(train_dir)
    print(f"✅ Dados carregados: {X_train.shape}, {y_train.shape}")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(1024, 512, 256),
            activation='relu',
            solver='adam',
            learning_rate_init=0.0003,
            validation_fraction=0.2,
            alpha=5e-3,
            batch_size=64,
            max_iter=3000,
            early_stopping=True,
            n_iter_no_change=50,
            random_state=42,
            verbose=True
        )
    )

    print("🚀 Iniciando treinamento...")
    model.fit(X_train, y_train_enc)
    print("✅ Treinamento concluído!")

    joblib.dump(model, "fer_mlp_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("💾 Modelo e encoder salvos!")

# =======================
# Execução em tempo real
# =======================
def realtime():
    print("📦 Carregando modelo...")
    model = joblib.load("fer_mlp_model.pkl")
    le = joblib.load("label_encoder.pkl")

    cap = cv2.VideoCapture(0)
    buffer_size = 10
    predictions_buffer = deque(maxlen=buffer_size)

    with mp_face.FaceMesh(max_num_faces=1) as face_mesh_rt:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            features = extract_features(frame)
            if features is not None:
                pred = model.predict([features])[0]
                predictions_buffer.append(pred)

                # Suavização pelo buffer com verificação segura
                if len(predictions_buffer) > 0:
                    result = mode(predictions_buffer, keepdims=True)
                    smoothed_pred = result.mode[0]
                    emotion = le.inverse_transform([smoothed_pred])[0]

                    cv2.putText(frame, emotion, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("FER - MediaPipe + MLP", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc para sair
                break

    cap.release()
    cv2.destroyAllWindows()

# =======================
# Teste em imagens
# =======================

def test_images(test_dir):
    print("📦 Carregando modelo...")
    model = joblib.load("fer_mlp_model.pkl")
    le = joblib.load("label_encoder.pkl")

    results = []

    for emotion in os.listdir(test_dir):
        emotion_path = os.path.join(test_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))
            features = extract_features(img)
            if features is None:
                results.append({
                    "image": img_name, 
                    "true_emotion": emotion, 
                    "predicted_emotion": "could_not_extract"
                })
                continue

            pred = model.predict([features])[0]
            predicted_emotion = le.inverse_transform([pred])[0]
            results.append({
                "image": img_name, 
                "true_emotion": emotion, 
                "predicted_emotion": predicted_emotion
            })

    # Salva resultados
    df = pd.DataFrame(results)
    df.to_csv("test_results.csv", index=False)
    print("✅ Predições salvas em test_results.csv")

    # Calcula acurácia
    df_valid = df[df["predicted_emotion"] != "could_not_extract"]
    accuracy = (df_valid["true_emotion"] == df_valid["predicted_emotion"]).mean()
    print(f"📊 Acurácia: {accuracy*100:.2f}%")

    # Mostra algumas predições
    print(df_valid.head(10))


# =======================
# Execução principal
# =======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Treina o modelo")
    parser.add_argument("--realtime", action="store_true", help="Executa webcam")
    parser.add_argument("--test", action="store_true", help="Testa imagens da pasta")  # <- adicionado
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.realtime:
        realtime()
    elif args.test: 
        test_images(test_dir)
    else:
        print("Use --train para treinar, --realtime para rodar a webcam ou --test para testar imagens.")

