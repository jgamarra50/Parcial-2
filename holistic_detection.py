import cv2
import mediapipe as mp
import numpy as np
import math
import os

# Configuración
CONFIRM_FRAMES = 3
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Cargar imágenes de referencia
REFERENCE_POSES = {}

def load_reference_poses():
    """Carga las imágenes de referencia del robot"""
    robot_folder = "robot"
    if not os.path.exists(robot_folder):
        print(f"Error: La carpeta '{robot_folder}' no existe")
        return False
    
    image_files = [f for f in os.listdir(robot_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(robot_folder, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            # Redimensionar para mostrar como miniatura
            img = cv2.resize(img, (150, 150))
            pose_name = os.path.splitext(img_file)[0]
            REFERENCE_POSES[pose_name] = img
            print(f"Cargada: {pose_name}")
    
    return len(REFERENCE_POSES) > 0

def angle_between(a, b, c):
    """Calcula el ángulo en el punto b entre los puntos a-b-c"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def calculate_horizontal_ratio(shoulder, elbow, wrist):
    """Calcula el ratio horizontal normalizado del brazo"""
    arm_len = np.linalg.norm(np.array(shoulder) - np.array(elbow)) + \
              np.linalg.norm(np.array(elbow) - np.array(wrist))
    hor_ratio = abs(wrist[0] - shoulder[0]) / max(1.0, arm_len)
    return hor_ratio

def detect_arm_state(shoulder, elbow, wrist, hip):
    """Detecta el estado del brazo: extended, up, down o neutral"""
    elbow_angle = angle_between(shoulder, elbow, wrist)
    hor_ratio = calculate_horizontal_ratio(shoulder, elbow, wrist)
    
    vx = wrist[0] - shoulder[0]
    vy = shoulder[1] - wrist[1]
    sw_angle = math.degrees(math.atan2(vy, vx))
    
    is_extended = elbow_angle > 150 and hor_ratio > 0.55
    is_up = wrist[1] < shoulder[1]
    is_down = wrist[1] > hip[1] or sw_angle < -60
    
    if is_extended:
        return "EXTENDED"
    elif is_up:
        return "UP"
    elif is_down:
        return "DOWN"
    else:
        return "NEUTRAL"

def match_pose(left_state, right_state):
    """Compara el estado actual con las poses de referencia"""
    pose_map = {
        ("UP", "UP"): "ambos_arriba",
        ("DOWN", "DOWN"): "ambos_abajo",
        ("EXTENDED", "EXTENDED"): "ambos_extendida",
        ("UP", "EXTENDED"): "izquierda_arriba_der_abajo",
        ("EXTENDED", "UP"): "derecha_arriba_izq_extendida",
        ("DOWN", "EXTENDED"): "izquierda_abajo_derecha_extendida",
        ("EXTENDED", "DOWN"): "derecha_extendida_izq_abajo",
        ("UP", "DOWN"): "izquierda_arriba_derecha_abajo",
        ("DOWN", "UP"): "izquierda_abajo_derecha_arriba",
    }
    
    key = (left_state, right_state)
    matched = pose_map.get(key, None)
    
    # Buscar coincidencia parcial en nombres de archivos
    if matched:
        for pose_name in REFERENCE_POSES.keys():
            if matched in pose_name.lower().replace("_", "").replace("-", ""):
                return pose_name
            # Búsqueda más flexible
            if left_state.lower() in pose_name.lower() and right_state.lower() in pose_name.lower():
                return pose_name
    
    return None

def main():
    # Cargar poses de referencia
    print("Cargando poses de referencia...")
    if not load_reference_poses():
        print("No se pudieron cargar las imágenes de referencia")
        return
    
    print(f"\nPoses cargadas: {list(REFERENCE_POSES.keys())}\n")
    
    # Configurar cámara
    print("="*50)
    print("OPCIONES DE ENTRADA:")
    print("1. Webcam del computador (si tiene)")
    print("2. Video pregrabado desde celular")
    print("="*50)
    
    option = input("\nSelecciona una opción (1 o 2): ").strip()
    
    is_video_file = False
    
    if option == "2":
        print("\n📱 CÓMO TRANSFERIR EL VIDEO:")
        print("  - WhatsApp Web (envíatelo y descárgalo)")
        print("  - Email")
        print("  - Google Drive / OneDrive")
        print("  - Bluetooth")
        print("\nGuarda el video en la carpeta 'Parcial 2'\n")
        
        video_path = input("Ingresa el nombre del video (ej: poses.mp4): ").strip()
        
        # Si solo puso el nombre, buscar en la carpeta actual
        if not os.path.isabs(video_path) and not os.path.exists(video_path):
            # Intentar encontrarlo en la carpeta actual
            if os.path.exists(video_path):
                pass
            else:
                print(f"⚠️  No se encontró '{video_path}' en la carpeta actual")
                print(f"Buscando en carpeta actual...")
                # Listar videos disponibles
                videos = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                if videos:
                    print(f"\nVideos disponibles:")
                    for i, v in enumerate(videos, 1):
                        print(f"  {i}. {v}")
                    return
        
        cap = cv2.VideoCapture(video_path)
        is_video_file = True
    else:
        print("\n📹 Intentando abrir webcam...")
        print("Si no funciona, prueba la opción 2 (video pregrabado)\n")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    print("\n✅ Cámara conectada!")
    print("\nControles:")
    print("  - Presiona 'Q' para salir")
    print("  - Haz las poses del robot para que se detecten\n")
    
    # Contadores de confirmación
    left_counters = {"EXTENDED": 0, "UP": 0, "DOWN": 0, "NEUTRAL": 0}
    right_counters = {"EXTENDED": 0, "UP": 0, "DOWN": 0, "NEUTRAL": 0}
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame")
                break
            
            # Espejo horizontal para mejor UX
            frame = cv2.flip(frame, 1)
            
            # Procesar
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w, _ = frame.shape
                
                # Extraer coordenadas
                left_shoulder = (landmarks[11].x * w, landmarks[11].y * h)
                left_elbow = (landmarks[13].x * w, landmarks[13].y * h)
                left_wrist = (landmarks[15].x * w, landmarks[15].y * h)
                left_hip = (landmarks[23].x * w, landmarks[23].y * h)
                
                right_shoulder = (landmarks[12].x * w, landmarks[12].y * h)
                right_elbow = (landmarks[14].x * w, landmarks[14].y * h)
                right_wrist = (landmarks[16].x * w, landmarks[16].y * h)
                right_hip = (landmarks[24].x * w, landmarks[24].y * h)
                
                # Detectar estados
                left_state = detect_arm_state(left_shoulder, left_elbow, left_wrist, left_hip)
                right_state = detect_arm_state(right_shoulder, right_elbow, right_wrist, right_hip)
                
                # Actualizar contadores
                for state in left_counters:
                    if state == left_state:
                        left_counters[state] = min(left_counters[state] + 1, CONFIRM_FRAMES)
                    else:
                        left_counters[state] = max(left_counters[state] - 1, 0)
                
                for state in right_counters:
                    if state == right_state:
                        right_counters[state] = min(right_counters[state] + 1, CONFIRM_FRAMES)
                    else:
                        right_counters[state] = max(right_counters[state] - 1, 0)
                
                # Estados confirmados
                left_confirmed = left_state if left_counters[left_state] >= CONFIRM_FRAMES else "..."
                right_confirmed = right_state if right_counters[right_state] >= CONFIRM_FRAMES else "..."
                
                # Buscar coincidencia de pose
                matched_pose = None
                if left_confirmed != "..." and right_confirmed != "...":
                    matched_pose = match_pose(left_confirmed, right_confirmed)
                
                # Dibujar landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Mostrar información
                cv2.putText(frame, f"Izq: {left_confirmed}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Der: {right_confirmed}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Si hay coincidencia, mostrar la imagen de referencia
                if matched_pose and matched_pose in REFERENCE_POSES:
                    cv2.putText(frame, f"POSE: {matched_pose}!", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    
                    # Mostrar miniatura de la pose detectada
                    ref_img = REFERENCE_POSES[matched_pose]
                    x_offset = w - 160
                    y_offset = 10
                    frame[y_offset:y_offset+150, x_offset:x_offset+150] = ref_img
            
            # Mostrar frame
            cv2.imshow('Detector de Poses - Presiona Q para salir', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Programa finalizado")

if __name__ == "__main__":
    main()