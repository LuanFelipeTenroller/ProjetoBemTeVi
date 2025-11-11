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
    """Extrai features geométricas da face."""
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
    landmarks = normalize_face(landmarks)

    idx = {
        "left_eye": 33, "right_eye": 263,
        "nose": 1, "mouth_left": 61, "mouth_right": 291,
        "chin": 199, "brow_left": 105, "brow_right": 334
    }

    # Distâncias
    d1 = distance(landmarks[idx["left_eye"]], landmarks[idx["right_eye"]])
    d2 = distance(landmarks[idx["nose"]], landmarks[idx["chin"]])
    d3 = distance(landmarks[idx["mouth_left"]], landmarks[idx["mouth_right"]])
    d4 = distance(landmarks[idx["brow_left"]], landmarks[idx["brow_right"]])

    # Ângulos
    a1 = angle(landmarks[idx["left_eye"]], landmarks[idx["nose"]], landmarks[idx["right_eye"]])
    a2 = angle(landmarks[idx["mouth_left"]], landmarks[idx["nose"]], landmarks[idx["mouth_right"]])
    a3 = angle(landmarks[idx["brow_left"]], landmarks[idx["nose"]], landmarks[idx["brow_right"]])

    return np.array([d1, d2, d3, d4, a1, a2, a3])

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
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            batch_size=16,
            max_iter=1000,
            alpha=0.001,
            verbose=True,
            random_state=42
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

                # Suavização pelo buffer
                smoothed_pred = mode(predictions_buffer)[0][0]
                emotion = le.inverse_transform([smoothed_pred])[0]

                cv2.putText(frame, emotion, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("FER - MediaPipe + MLP", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc para sair
                break

    cap.release()
    cv2.destroyAllWindows()

# =======================
# Execução principal
# =======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Treina o modelo")
    parser.add_argument("--realtime", action="store_true", help="Executa webcam")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.realtime:
        realtime()
    else:
        print("Use --train para treinar ou --realtime para rodar a webcam.")
