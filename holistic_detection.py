#!/usr/bin/env python3
"""
Robot Virtual - Control por Detección de Poses con MediaPipe
Sistema de detección de poses de brazos en tiempo real con filtrado de ruido
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
from pathlib import Path
from collections import deque

# Configuración de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración de parámetros
CONFIRM_FRAMES = 4  # Frames necesarios para confirmar cambio de estado
SMOOTHING_WINDOW = 5  # Ventana para suavizado de ángulos

# Umbrales de detección
EXTENDED_ANGLE_THRESHOLD = 150  # Grados para considerar brazo extendido
HORIZONTAL_RATIO_THRESHOLD = 0.55  # Ratio para brazo horizontal
UP_ANGLE_THRESHOLD = 60  # Ángulo para considerar brazo arriba
DOWN_ANGLE_THRESHOLD = -60  # Ángulo para considerar brazo abajo


class ArmState:
    """Estados posibles de un brazo"""
    DOWN = "down"
    UP = "up"
    EXTENDED = "extended"


class SmoothingFilter:
    """Filtro de suavizado exponencial para valores numéricos"""
    
    def __init__(self, window_size=SMOOTHING_WINDOW):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def update(self, value):
        """Actualiza con nuevo valor y retorna el promedio"""
        self.values.append(value)
        return np.median(self.values) if len(self.values) > 0 else value


class StateConfirmation:
    """Mecanismo de histeresis para confirmar cambios de estado"""
    
    def __init__(self, confirm_frames=CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self.current_state = None
        self.candidate_state = None
        self.counter = 0
    
    def update(self, new_state):
        """Actualiza estado con confirmación por frames"""
        if new_state == self.current_state:
            # Estado se mantiene
            self.counter = 0
            return self.current_state
        
        if new_state == self.candidate_state:
            # Incrementar contador para candidato
            self.counter += 1
            if self.counter >= self.confirm_frames:
                # Confirmar nuevo estado
                self.current_state = new_state
                self.counter = 0
                self.candidate_state = None
        else:
            # Nuevo candidato
            self.candidate_state = new_state
            self.counter = 1
        
        return self.current_state if self.current_state else new_state


class RobotController:
    """Controlador principal del robot virtual"""
    
    def __init__(self, robot_images_path='robot'):
        self.robot_images_path = Path(robot_images_path)
        self.images = self._load_robot_images()
        self.current_state = "both_down"
        
        # Filtros de suavizado para cada brazo
        self.left_angle_filter = SmoothingFilter()
        self.right_angle_filter = SmoothingFilter()
        self.left_ratio_filter = SmoothingFilter()
        self.right_ratio_filter = SmoothingFilter()
        
        # Confirmación de estados
        self.state_confirmation = StateConfirmation()
    
    def _load_robot_images(self):
        """Carga todas las imágenes del robot"""
        images = {}
        image_names = [
            'ambos_abajo',
            'ambos_arriba',
            'ambos_extendida',
            'derecha_arriba_izq_abajo',
            'derecha_arriba_izq_extendida',
            'derecha_extendida_izq_abajo',
            'izquierda_arriba_der_abajo',
            'izquierda_arriba_der_extendida',
            'izquierda_extendida_der_abajo'
        ]
        
        for name in image_names:
            img_path = self.robot_images_path / f"{name}.png"
            if img_path.exists():
                images[name] = cv2.imread(str(img_path))
            else:
                # Imagen placeholder si no existe
                images[name] = self._create_placeholder(name)
        
        return images
    
    def _create_placeholder(self, name):
        """Crea imagen placeholder"""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        cv2.putText(img, name.replace('_', ' ').upper(), 
                   (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 0), 2)
        return img
    
    @staticmethod
    def angle_between(a, b, c):
        """
        Calcula el ángulo en el punto b entre tres puntos a-b-c
        Usa producto punto y arccos para el cálculo
        
        Args:
            a, b, c: Puntos con coordenadas [x, y]
        
        Returns:
            Ángulo en grados
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        # Producto punto y normas
        dot_product = np.dot(ba, bc)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        # Evitar división por cero
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        
        # cos(θ) = (a·b) / (||a|| ||b||)
        cosang = dot_product / (norm_ba * norm_bc)
        
        # Clip para evitar errores numéricos en arccos
        cosang = np.clip(cosang, -1.0, 1.0)
        
        # Convertir a grados
        angle = np.degrees(np.arccos(cosang))
        
        return angle
    
    @staticmethod
    def calculate_direction_angle(shoulder, wrist):
        """
        Calcula el ángulo de dirección hombro→muñeca
        
        Args:
            shoulder, wrist: Puntos [x, y]
        
        Returns:
            Ángulo en grados (0°=derecha, 90°=arriba, -90°=abajo)
        """
        vx = wrist[0] - shoulder[0]
        vy = shoulder[1] - wrist[1]  # Invertir Y (coords de píxel)
        
        angle = math.degrees(math.atan2(vy, vx))
        return angle
    
    @staticmethod
    def calculate_horizontal_ratio(shoulder, elbow, wrist):
        """
        Calcula la ratio horizontal normalizada del brazo
        
        Args:
            shoulder, elbow, wrist: Puntos [x, y]
        
        Returns:
            Ratio horizontal (0 a 1+)
        """
        # Longitud total del brazo
        dist_shoulder_elbow = np.linalg.norm(np.array(shoulder) - np.array(elbow))
        dist_elbow_wrist = np.linalg.norm(np.array(elbow) - np.array(wrist))
        arm_length = dist_shoulder_elbow + dist_elbow_wrist
        
        if arm_length == 0:
            return 0.0
        
        # Componente horizontal
        horizontal_dist = abs(wrist[0] - shoulder[0])
        
        # Ratio normalizada
        ratio = horizontal_dist / arm_length
        
        return ratio
    
    def detect_arm_state(self, shoulder, elbow, wrist, hip, is_left=True):
        """
        Detecta el estado de un brazo usando múltiples heurísticas
        
        Args:
            shoulder, elbow, wrist, hip: Puntos [x, y]
            is_left: Si es el brazo izquierdo
        
        Returns:
            Estado del brazo (ArmState)
        """
        # Calcular ángulo de codo
        elbow_angle = self.angle_between(shoulder, elbow, wrist)
        
        # Calcular ángulo de dirección
        direction_angle = self.calculate_direction_angle(shoulder, wrist)
        
        # Calcular ratio horizontal
        horizontal_ratio = self.calculate_horizontal_ratio(shoulder, elbow, wrist)
        
        # Aplicar suavizado
        if is_left:
            elbow_angle = self.left_angle_filter.update(elbow_angle)
            horizontal_ratio = self.left_ratio_filter.update(horizontal_ratio)
        else:
            elbow_angle = self.right_angle_filter.update(elbow_angle)
            horizontal_ratio = self.right_ratio_filter.update(horizontal_ratio)
        
        # Heurística de detección
        # 1. EXTENDED: brazo extendido horizontalmente
        if elbow_angle > EXTENDED_ANGLE_THRESHOLD and horizontal_ratio > HORIZONTAL_RATIO_THRESHOLD:
            return ArmState.EXTENDED
        
        # 2. UP: muñeca arriba del hombro y ángulo de dirección hacia arriba
        if wrist[1] < shoulder[1] and direction_angle > UP_ANGLE_THRESHOLD:
            return ArmState.UP
        
        # 3. DOWN: muñeca abajo de la cadera o ángulo hacia abajo
        if wrist[1] > hip[1] or direction_angle < DOWN_ANGLE_THRESHOLD:
            return ArmState.DOWN
        
        # Por defecto, DOWN
        return ArmState.DOWN
    
    def map_states_to_image_name(self, left_state, right_state):
        """
        Mapea estados de brazos a nombre de imagen
        
        Args:
            left_state, right_state: Estados de brazos (ArmState)
        
        Returns:
            Nombre de la imagen del robot
        """
        state_map = {
            (ArmState.DOWN, ArmState.DOWN): 'ambos_abajo',
            (ArmState.UP, ArmState.UP): 'ambos_arriba',
            (ArmState.EXTENDED, ArmState.EXTENDED): 'ambos_extendida',
            (ArmState.DOWN, ArmState.UP): 'derecha_arriba_izq_abajo',
            (ArmState.EXTENDED, ArmState.UP): 'derecha_arriba_izq_extendida',
            (ArmState.DOWN, ArmState.EXTENDED): 'derecha_extendida_izq_abajo',
            (ArmState.UP, ArmState.DOWN): 'izquierda_arriba_der_abajo',
            (ArmState.UP, ArmState.EXTENDED): 'izquierda_arriba_der_extendida',
            (ArmState.EXTENDED, ArmState.DOWN): 'izquierda_extendida_der_abajo'
        }
        
        return state_map.get((left_state, right_state), 'ambos_abajo')
    
    def process_pose(self, landmarks, image_width, image_height):
        """
        Procesa los landmarks de MediaPipe y determina el estado del robot
        
        Args:
            landmarks: pose_landmarks de MediaPipe
            image_width, image_height: Dimensiones de la imagen
        
        Returns:
            Nombre del estado del robot
        """
        # Extraer puntos relevantes y convertir a píxeles
        def to_pixel(landmark):
            return [int(landmark.x * image_width), 
                   int(landmark.y * image_height)]
        
        # Hombros
        left_shoulder = to_pixel(landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = to_pixel(landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER])
        
        # Codos
        left_elbow = to_pixel(landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW])
        right_elbow = to_pixel(landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW])
        
        # Muñecas
        left_wrist = to_pixel(landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST])
        right_wrist = to_pixel(landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST])
        
        # Caderas (para referencia)
        left_hip = to_pixel(landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP])
        right_hip = to_pixel(landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP])
        
        # Detectar estado de cada brazo
        left_state = self.detect_arm_state(left_shoulder, left_elbow, left_wrist, left_hip, is_left=True)
        right_state = self.detect_arm_state(right_shoulder, right_elbow, right_wrist, right_hip, is_left=False)
        
        # Mapear a nombre de imagen
        new_state = self.map_states_to_image_name(left_state, right_state)
        
        # Aplicar confirmación de estado (histeresis)
        confirmed_state = self.state_confirmation.update(new_state)
        
        if confirmed_state:
            self.current_state = confirmed_state
        
        return self.current_state, left_state, right_state
    
    def get_robot_image(self):
        """Obtiene la imagen actual del robot"""
        return self.images.get(self.current_state, self.images['ambos_abajo']).copy()


def draw_overlay(image, state, left_state, right_state, fps=0):
    """Dibuja información sobre la imagen"""
    overlay = image.copy()
    
    # Fondo semitransparente
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Texto
    cv2.putText(image, f"Estado: {state}", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, f"Izq: {left_state}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"Der: {right_state}", (20, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"FPS: {fps:.1f}", (20, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def main():
    """Función principal"""
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Robot Virtual con MediaPipe')
    parser.add_argument('--video', type=str, help='Ruta a archivo de video (opcional)')
    args = parser.parse_args()
    
    # Inicializar controlador
    controller = RobotController('robot')
    
    # Configurar captura de video
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Usando video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Usando cámara por defecto")
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara o video")
        return
    
    print("\n" + "="*60)
    print(" ROBOT VIRTUAL - CONTROL POR POSES")
    print("="*60)
    print("\nInstrucciones:")
    print("  - Mueve tus brazos para controlar el robot")
    print("  - El robot imitará tus poses")
    print("  - Presiona 'q' para salir")
    print("  - Presiona 'r' para resetear filtros")
    print("="*60 + "\n")
    
    # Inicializar MediaPipe Holistic
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        # Variables para FPS
        fps = 0
        frame_count = 0
        import time
        start_time = time.time()
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                if args.video:
                    # Reiniciar video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Error al leer frame")
                    break
            
            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Obtener dimensiones
            height, width = frame.shape[:2]
            
            # Convertir BGR a RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Procesar con MediaPipe
            results = holistic.process(image_rgb)
            
            # Convertir de vuelta a BGR
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Variables de estado
            state = "both_down"
            left_state = "down"
            right_state = "down"
            
            # Si hay landmarks de pose
            if results.pose_landmarks:
                # Dibujar landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Procesar pose y obtener estado
                state, left_state, right_state = controller.process_pose(
                    results.pose_landmarks, width, height
                )
            
            # Calcular FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = end_time
            
            # Dibujar overlay con información
            draw_overlay(image, state, left_state, right_state, fps)
            
            # Obtener imagen del robot
            robot_img = controller.get_robot_image()
            
            # Añadir título a la imagen del robot
            cv2.putText(robot_img, f"Estado: {state}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Redimensionar imagen del robot
            robot_display = cv2.resize(robot_img, (500, 500))
            
            # Mostrar ventanas
            cv2.imshow('Camara - Deteccion de Poses', image)
            cv2.imshow('Robot Virtual', robot_display)
            
            # Manejar teclas
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Resetear filtros
                controller.left_angle_filter = SmoothingFilter()
                controller.right_angle_filter = SmoothingFilter()
                controller.left_ratio_filter = SmoothingFilter()
                controller.right_ratio_filter = SmoothingFilter()
                controller.state_confirmation = StateConfirmation()
                print("Filtros reseteados")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("\n¡Aplicación cerrada!")


if __name__ == "__main__":
    main()
