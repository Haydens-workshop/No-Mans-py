# utils.py

import cv2
import numpy as np
import pyautogui
import logging
import os
from datetime import datetime
import pytesseract
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class VideoRecorder:
    def __init__(self, width=1280, height=720, fps=30.0, filename="output.mp4"):
        self.width = width
        self.height = height
        self.fps = fps
        self.filename = filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def add_frame(self, frame):
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        self.out.write(frame)

    def close(self):
        self.out.release()

def setup_logger(log_level=logging.INFO):
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=f'nms_ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def create_video_recorder(env, output_folder="videos"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_folder, f"nms_ai_video_{timestamp}.mp4")
    return VideoRecorder(filename=filename)

def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def process_image(image, target_size=(84, 84)):
    resized = cv2.resize(image, target_size)
    normalized = resized / 255.0
    return normalized

def detect_text_on_screen(image, keywords):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(Image.fromarray(gray))
    results = {}
    for keyword in keywords:
        if keyword.lower() in text.lower():
            # Extract the value associated with the keyword
            # This is a simple implementation and might need to be adjusted
            value = text.lower().split(keyword.lower())[-1].split()[0]
            try:
                results[keyword] = int(value)
            except ValueError:
                results[keyword] = value
    return results

def detect_objects_on_screen(image, object_types):
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Prepare the image
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)

    # Perform object detection
    with torch.no_grad():
        prediction = model([img_tensor])

    # Process the results
    results = {}
    for label, score, box in zip(prediction[0]['labels'], prediction[0]['scores'], prediction[0]['boxes']):
        if score > 0.5:  # Consider only detections with confidence > 0.5
            label_name = object_types.get(label.item(), 'unknown')
            if label_name not in results:
                results[label_name] = []
            results[label_name].append({
                'score': score.item(),
                'box': box.tolist()
            })

    return results

def calculate_distance(pos1, pos2):
    return np.sqrt(sum((np.array(pos1) - np.array(pos2)) ** 2))

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

def get_direction_vector(start, end):
    return normalize_vector(np.array(end) - np.array(start))

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def apply_fog_of_war(image, explored_positions, visibility_radius):
    h, w = image.shape[:2]
    fog = np.zeros((h, w), dtype=np.uint8)
    for pos in explored_positions:
        mask = create_circular_mask(h, w, pos, visibility_radius)
        fog[mask] = 255
    return cv2.bitwise_and(image, image, mask=fog)

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

# Color detection utilities
def detect_color_range(image, lower_color, upper_color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return cv2.countNonZero(mask)

def is_color_present(image, color, threshold=1000):
    lower = np.array([color[0] - 10, 100, 100])
    upper = np.array([color[0] + 10, 255, 255])
    count = detect_color_range(image, lower, upper)
    return count > threshold

# UI element detection
def detect_ui_element(screenshot, template_path, threshold=0.8):
    template = cv2.imread(template_path, 0)
    gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return list(zip(*loc[::-1]))  # Returns list of (x, y) tuples where the UI element was found

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.frame_count = 0
        self.total_reward = 0

    def update(self, reward):
        self.frame_count += 1
        self.total_reward += reward

    def get_stats(self):
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        return {
            'fps': self.frame_count / elapsed_time,
            'avg_reward': self.total_reward / self.frame_count if self.frame_count > 0 else 0,
            'total_frames': self.frame_count,
            'elapsed_time': elapsed_time
        }

# Environment wrapper for frame skipping
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(FrameSkip, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info