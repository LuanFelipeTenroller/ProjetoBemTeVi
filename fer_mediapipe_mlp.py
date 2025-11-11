import cv2
import mediapipe as mp
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# Caminhos dos dados
train_dir = "data/fer2013/train"
test_dir = "data/fer2013/test"

# Emoções
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

# Função para extrair landmarks da face
def extract_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return None

# Função para carregar dataset
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
            landmarks = extract_landmarks(img)
            if landmarks is not None:
                X.append(landmarks)
                y.append(emotion)
    return np.array(X), np.array(y)

# Treinamento
def train_model():
    print("Carregando dados de treino...")
    X_train, y_train = load_data(train_dir)
    print(f"Treino: {X_train.shape}, {y_train.shape}")

    # Encode das labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # Cria MLP
    model = MLPClassifier(hidden_layer_sizes=(1024, 1024),
                          activation='relu', solver='adam',
                          batch_size=32, max_iter=50, verbose=True)

    # Treina
    model.fit(X_train, y_train_enc)
    print("Treinamento concluído!")

    # Salva modelo e encoder
    joblib.dump(model, "fer_mlp_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("Modelo salvo!")

# Reconhecimento em tempo real
def realtime():
    print("Carregando modelo...")
    model = joblib.load("fer_mlp_model.pkl")
    le = joblib.load("label_encoder.pkl")

    cap = cv2.VideoCapture(0)
    with mp_face.FaceMesh(max_num_faces=1) as face_mesh_rt:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = extract_landmarks(frame)
            if landmarks is not None:
                pred = model.predict([landmarks])
                emotion = le.inverse_transform(pred)[0]
                cv2.putText(frame, emotion, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("FER - MediaPipe + MLP", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc para sair
                break

    cap.release()
    cv2.destroyAllWindows()

# Main
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
        print("Use --train ou --realtime")
