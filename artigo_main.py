#!/usr/bin/env python3
# artigo_main.py
# Implementação fiel ao artigo "Real-Time Facial Expression Recognition using Facial Landmarks and Neural Networks"
# Requer: dlib shape predictor (shape_predictor_68_face_landmarks.dat) e dataset CK+ organizado por pastas por emoção.

import os
import time
import math
import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------------------
# CONFIGURAÇÕES
# ---------------------------
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CKPLUS_DIR = "data/CKPlus"
EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (480, 480)
RANDOM_STATE = 42
NUM_CLASSES = len(EMOTIONS)

# ---------------------------
# VERIFICA DEPENDÊNCIAS
# ---------------------------
for lib in ["cv2", "dlib", "tensorflow"]:
    if lib not in globals():
        print(f"⚠️  Dependência ausente: {lib}. Instale com 'pip install {lib}'.")

# ---------------------------
# SELECIONA GPU AUTOMATICAMENTE
# ---------------------------
if tf.config.list_physical_devices('GPU'):
    DEVICE = "/GPU:0"
    print("✅ GPU detectada. Usando acelerador CUDA.")
else:
    DEVICE = "/CPU:0"
    print("⚠️ GPU não detectada. Usando CPU.")

# ---------------------------
# Inicializa detector/shape predictor
# ---------------------------
detector = dlib.get_frontal_face_detector()
if not os.path.exists(SHAPE_PREDICTOR_PATH):
    raise FileNotFoundError(f"shape predictor not found: {SHAPE_PREDICTOR_PATH}")
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# ---------------------------
# UTILS
# ---------------------------
def to_np(shape):
    coords = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        part = shape.part(i)
        coords[i] = (part.x, part.y)
    return coords

def rotate_points(points, center, angle_rad):
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    shifted = points - center
    rotated = shifted.dot(R.T) + center
    return rotated

def normalize_roll_and_yaw(landmarks):
    left_eye = landmarks[42:48].mean(axis=0)
    right_eye = landmarks[36:42].mean(axis=0)
    eyes_center = (left_eye + right_eye) / 2.0

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_rad = math.atan2(dy, dx)
    rotated = rotate_points(landmarks, eyes_center, -angle_rad)

    pairs = [
        (0,16),(1,15),(2,14),(3,13),(4,12),(5,11),(6,10),(7,9),
        (17,26),(18,25),(19,24),(20,23),(21,22),
        (36,45),(37,44),(38,43),(39,42),(40,47),(41,46),
        (31,35),(32,34),
        (48,54),(49,53),(50,52),
        (61,63),(60,64),(67,65)
    ]
    sym = rotated.copy()
    for a,b in pairs:
        avg = (rotated[a] + rotated[b]) / 2.0
        sym[a] = avg
        sym[b] = avg
    return sym

def angle_between(a, b, c):
    va, vc = a - b, c - b
    denom = (np.linalg.norm(va) * np.linalg.norm(vc)) + 1e-8
    cosang = np.dot(va, vc) / denom
    return float(math.acos(np.clip(cosang, -1.0, 1.0)))

# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
def extract_features_from_landmarks(landmarks, gray_image):
    lm = normalize_roll_and_yaw(landmarks)

    right_eyebrow_center = lm[17:22].mean(axis=0)
    right_eye_center = lm[36:42].mean(axis=0)
    left_eyebrow_center = lm[22:27].mean(axis=0)
    left_eye_center = lm[42:48].mean(axis=0)
    mouth_outer = lm[48:60]
    mouth_top = np.mean([lm[50], lm[51], lm[52]], axis=0)
    mouth_bottom = np.mean([lm[57], lm[58], lm[59]], axis=0)
    nose_tip, chin = lm[30], lm[8]
    mouth_center = mouth_outer.mean(axis=0)

    geom_angles = [
        angle_between(right_eyebrow_center, nose_tip, left_eyebrow_center),
        angle_between(right_eye_center, nose_tip, left_eye_center),
        angle_between(mouth_top, nose_tip, mouth_center),
        angle_between(mouth_bottom, mouth_top, nose_tip),
        angle_between(chin, mouth_center, nose_tip),
    ]

    sobel = np.abs(cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3))
    sobel = np.clip(sobel, 0, 255)
    H, W = gray_image.shape

    def region_density(pts):
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.int32(pts)], 1)
        vals = sobel[mask == 1]
        return float(np.mean(vals) / 255.0) if vals.size else 0.0

    regions = [
        [lm[39], lm[42], lm[27], lm[28]],  # between eyes
        [lm[36], lm[39], lm[41], lm[40]],  # right eye
        [lm[42], lm[45], lm[47], lm[46]],  # left eye
        [lm[36], lm[41], lm[31], lm[48]],  # right cheek
        [lm[45], lm[46], lm[35], lm[54]],  # left cheek
    ]
    texture = [region_density(r) for r in regions]
    return np.concatenate([geom_angles, texture], dtype=np.float32)

# ---------------------------
# EXTRATOR DE FEATURES POR IMAGEM
# ---------------------------
def image_to_feature_vector(image_bgr):
    """
    Converte imagem já recortada de rosto (CK+) em vetor simples de características.
    Evita usar dlib para detectar rostos pequenos (48x48).
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        return None

    # Garante formato e tipo
    if image_bgr.dtype != np.uint8:
        image_bgr = cv2.convertScaleAbs(image_bgr)
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    # Redimensiona (opcional)
    img = cv2.resize(image_bgr, (48, 48))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normaliza e achata (gera vetor de 2304 features = 48x48)
    feat = gray.flatten().astype(np.float32) / 255.0
    return feat

# ---------------------------
# CARREGAR CK+
# ---------------------------
def load_ckplus_features(ck_dir):
    ck_dir = ck_dir.capitalize() if not os.path.exists(ck_dir) else ck_dir
    X, y = [], []
    print("🔹 Extraindo features do CK+ ...")
    for emo in EMOTIONS:
        folder = os.path.join(ck_dir, emo)
        if not os.path.isdir(folder):
            print(f"⚠️ Pasta ausente: {folder}")
            continue
        files = sorted(glob(os.path.join(folder, "*.*")))
        for f in tqdm(files, desc=f"{emo}", leave=False):
            img = cv2.imread(f)
            if img is None:
                continue
            feat = image_to_feature_vector(img)
            if feat is not None:
                X.append(feat)
                y.append(emo)
    X, y = np.array(X, np.float32), np.array(y)
    print(f"✅ Extraídos {X.shape[0]} exemplos ({X.shape[1]} features cada).")
    return X, y

# ---------------------------
# MLP conforme artigo
# ---------------------------
def build_mlp(num_features, num_classes):
    model = Sequential([
        Dense(1024, input_shape=(num_features,), kernel_initializer='glorot_uniform'),
        BatchNormalization(momentum=0.99),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(1024, kernel_initializer='glorot_uniform'),
        BatchNormalization(momentum=0.99),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------
# TREINAMENTO
# ---------------------------
def train_and_evaluate(X, y, folds=5, epochs=200, batch_size=16):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    os.makedirs("saved_models", exist_ok=True)

    accs, all_true, all_pred = [], [], []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    for i, (train, test) in enumerate(skf.split(X, y_enc), 1):
        print(f"\n===== Fold {i}/{folds} =====")
        model = build_mlp(X.shape[1], NUM_CLASSES)
        cb = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8),
            ModelCheckpoint(f"saved_models/model_fold{i}.h5", save_best_only=True, monitor='val_loss')
        ]
        model.fit(X[train], y_enc[train],
                  validation_data=(X[test], y_enc[test]),
                  epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=1)
        preds = np.argmax(model.predict(X[test]), axis=1)
        acc = np.mean(preds == y_enc[test])
        accs.append(acc)
        all_true.extend(y_enc[test])
        all_pred.extend(preds)
        print(f"Fold {i} accuracy: {acc*100:.2f}%")

    accs = np.array(accs)
    mean, std = accs.mean(), accs.std()
    ci = 1.96 * std / math.sqrt(folds)
    print(f"\n✅ Média: {mean*100:.2f}% ± {ci*100:.2f}% (95% CI)")
    pd.DataFrame({'fold_acc': accs}).to_csv("fold_accuracies.csv", index=False)
    pd.to_pickle(le, "label_encoder.pkl")

# ---------------------------
# REAL-TIME
# ---------------------------
def realtime_predict(model_path, encoder_path="label_encoder.pkl"):
    model = tf.keras.models.load_model(model_path)
    le = pd.read_pickle(encoder_path) if os.path.exists(encoder_path) else None
    classes = le.classes_ if le else EMOTIONS

    cap = cv2.VideoCapture(0)
    prev = time.time()
    fps_avg = 0
    print("🎥 Pressione ESC para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feat = image_to_feature_vector(frame)
        if feat is not None:
            probs = model.predict(np.expand_dims(feat, 0))[0]
            idx, conf = np.argmax(probs), np.max(probs)
            label = f"{classes[idx].upper()} ({conf*100:.1f}%)"
            color = (0,255,0) if conf>0.6 else (0,165,255)
            cv2.putText(frame, label, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "Sem rosto detectado", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        now = time.time()
        fps = 1/(now-prev+1e-8)
        fps_avg = fps_avg*0.9 + fps*0.1
        prev = now
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (frame.shape[1]-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("FER (Article-based, Dlib+MLP)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--model", type=str, default="saved_models/model_fold1.h5")
    args = parser.parse_args()

    if args.extract:
        X, y = load_ckplus_features(CKPLUS_DIR)
        np.save("X_features.npy", X)
        np.save("y_labels.npy", y)
    elif args.train:
        X = np.load("X_features.npy")
        y = np.load("y_labels.npy", allow_pickle=True)
        train_and_evaluate(X, y)
    elif args.realtime:
        realtime_predict(args.model)
    else:
        print("Uso:")
        print("  python artigo_main.py --extract")
        print("  python artigo_main.py --train")
        print("  python artigo_main.py --realtime --model saved_models/model_fold1.h5")
