import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import datetime
import os
import collections

# Pastikan folder logs dan models ada
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Cek apakah model ada, jika tidak gunakan model alternatif atau tampilkan pesan
model_path = 'models/sign_language_mobilenetv2_regularized.h5'
if not os.path.exists(model_path):
    print(f"Model tidak ditemukan di {model_path}")
    model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
    if model_files:
        alternative_model = os.path.join('models', model_files[0])
        print(f"Menggunakan model alternatif: {alternative_model}")
        model_path = alternative_model
    else:
        raise FileNotFoundError(f"Tidak ada model .h5 yang ditemukan di folder 'models'. Silakan latih model terlebih dahulu.")

# Load trained CNN model
print(f"Loading model dari: {model_path}")
model = load_model(model_path)
print("Model berhasil dimuat!")

# Label mapping A-Z
labels = {i: chr(i+65) for i in range(26)}

# Init MediaPipe Hands dengan parameter yang dioptimalkan untuk deteksi isyarat tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Fokus pada satu tangan untuk mengurangi kesalahan
    min_detection_confidence=0.6,  # Lebih tinggi untuk mencegah deteksi false positive
    min_tracking_confidence=0.6    # Lebih tinggi untuk mencegah tracking yang tidak stabil
)

# Log hasil prediksi
def log_prediction(predicted_char, confidence):
    with open('logs/detection_log.txt', 'a') as log_file:
        log_file.write(f'{datetime.datetime.now()} - {predicted_char} (Confidence: {confidence:.2f}%)\n')

# Fungsi preprocessor yang konsisten dengan pelatihan
def preprocess_for_model(img):
    # Gunakan preprocessing yang sama dengan training
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalisasi sederhana seperti di train_cnn.py
    return img

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera. Periksa apakah kamera terhubung dengan benar.")
    exit()
else:
    print("Kamera berhasil dibuka.")

# Variabel untuk stabilisasi prediksi
prediction_queue = collections.deque(maxlen=10)  # Menyimpan 10 prediksi terakhir
stable_prediction = None
stable_confidence = 0
stable_count = 0
stability_threshold = 3  # Jumlah minimum prediksi yang sama berturut-turut
no_hand_counter = 0      # Menghitung berapa frame tanpa tangan terdeteksi
display_cooldown = 0     # Cooldown untuk display prediksi

# Statistik prediksi
prediction_count = {}
min_confidence = 35  # Turunkan threshold confidence

# Mode debug untuk melihat confidence setiap prediksi
debug_mode = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera.")
        break

    h, w, _ = frame.shape

    # Flip frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)
    
    # Copy frame untuk display
    display_frame = frame.copy()
    
    # Process frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_prediction = None
    current_confidence = 0
    
    # Handle kasus tidak ada tangan terdeteksi
    if not results.multi_hand_landmarks:
        no_hand_counter += 1
        # Setelah 15 frame tanpa tangan, reset stabilitas
        if no_hand_counter > 15:
            prediction_queue.clear()
            if display_cooldown <= 0:
                stable_prediction = None
            else:
                display_cooldown -= 1
    else:
        no_hand_counter = 0  
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmarks MediaPipe
            mp.solutions.drawing_utils.draw_landmarks(
                display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Hitung bounding box dengan padding
            x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
            y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
            x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
            y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)
            
            # Tambahkan padding (25% untuk memastikan seluruh tangan terlihat)
            padding_x = int((x_max - x_min) * 0.25)
            padding_y = int((y_max - y_min) * 0.25)
            
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(w, x_max + padding_x)
            y_max = min(h, y_max + padding_y)
            
            # Verifikasi dimensi valid
            if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                continue
                
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.shape[0] <= 0 or hand_img.shape[1] <= 0:
                continue
            
            # Preprocessing yang konsisten dengan training
            hand_img_processed = preprocess_for_model(hand_img)
            hand_img_array = np.expand_dims(hand_img_processed, axis=0)
            
            # Prediksi dengan model
            predictions = model.predict(hand_img_array, verbose=0)
            predicted_idx = np.argmax(predictions)
            confidence = predictions[0][predicted_idx] * 100
            predicted_char = labels[predicted_idx]
            
            current_prediction = predicted_char
            current_confidence = confidence
            
            # Tampilkan bounding box dan label
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{predicted_char} ({confidence:.1f}%)", 
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Simpan gambar tangan yang terdeteksi untuk debugging (opsional)
            if debug_mode and confidence > min_confidence:
                debug_dir = 'debug_images'
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%H%M%S-%f")
                cv2.imwrite(f'{debug_dir}/hand_{predicted_char}_{confidence:.1f}_{timestamp}.jpg', hand_img)
    
    # Update prediction queue jika ada prediksi baru
    if current_prediction and current_confidence > min_confidence:
        prediction_queue.append((current_prediction, current_confidence))
    
    # Analisis queue untuk stabilisasi
    if prediction_queue:
        # Hitung frekuensi setiap prediksi dalam queue
        pred_counter = collections.Counter([p[0] for p in prediction_queue])
        most_common = pred_counter.most_common(1)[0]  # (char, count)
        
        # Jika prediksi paling umum muncul cukup sering, anggap stabil
        if most_common[1] >= stability_threshold:
            new_stable_pred = most_common[0]
            # Hitung confidence rata-rata untuk prediksi ini
            avg_confidence = np.mean([conf for pred, conf in prediction_queue if pred == new_stable_pred])
            
            # Hanya update jika prediksi berubah atau confidence berubah signifikan
            if new_stable_pred != stable_prediction or abs(avg_confidence - stable_confidence) > 10:
                stable_prediction = new_stable_pred
                stable_confidence = avg_confidence
                stable_count = most_common[1]
                display_cooldown = 20  # Tampilkan huruf setidaknya selama 20 frame
                
                # Log prediksi stabil
                log_prediction(stable_prediction, stable_confidence)
                
                # Update statistik
                if stable_prediction in prediction_count:
                    prediction_count[stable_prediction] += 1
                else:
                    prediction_count[stable_prediction] = 1
    
    # Tampilkan stable prediction jika ada
    if stable_prediction and display_cooldown > 0:
        # Tampilkan predicted char besar di tengah layar dengan background semi-transparan
        # untuk meningkatkan keterbacaan
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (w//2 - 70, h//2 - 70), (w//2 + 70, h//2 + 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
        cv2.putText(display_frame, stable_prediction, 
                   (w//2 - 50, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 5)
        
        display_cooldown -= 1
    
    # Tampilkan statistik deteksi di pojok kanan atas
    y_pos = 30
    cv2.putText(display_frame, "Detected Letters:", (w - 200, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    y_pos += 20
    
    for char, count in sorted(prediction_count.items()):
        cv2.putText(display_frame, f"{char}: {count}", (w - 200, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_pos += 20
        if y_pos > h - 20:  # Prevent text from going off screen
            break
    
    # Tampilkan mode dan status
    if debug_mode:
        confidence_info = f"Current: {current_prediction}@{current_confidence:.1f}%" if current_prediction else "No detection"
        stability_info = f"Stable: {stable_prediction}@{stable_confidence:.1f}% ({stable_count}/{stability_threshold})" if stable_prediction else "No stable prediction"
        
        cv2.putText(display_frame, confidence_info, (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display_frame, stability_info, (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Tambahkan petunjuk keluar dan reset
    cv2.putText(display_frame, "Press 'q' to quit, 'r' to reset, 'd' for debug", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Sign Language Translator', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Aplikasi ditutup oleh pengguna.")
        break
    elif key == ord('r'):
        # Reset prediction count
        prediction_count = {}
        prediction_queue.clear()
        stable_prediction = None
        stable_confidence = 0
        stable_count = 0
        print("Statistik deteksi di-reset.")
    elif key == ord('d'):
        # Toggle debug mode
        debug_mode = not debug_mode
        print(f"Debug mode: {'On' if debug_mode else 'Off'}")
    elif key == ord('+'):
        # Meningkatkan threshold stabilitas
        stability_threshold = min(stability_threshold + 1, 10)
        print(f"Stability threshold: {stability_threshold}")
    elif key == ord('-'):
        # Menurunkan threshold stabilitas
        stability_threshold = max(stability_threshold - 1, 2)
        print(f"Stability threshold: {stability_threshold}")

cap.release()
cv2.destroyAllWindows()
print("Aplikasi berhasil ditutup.")