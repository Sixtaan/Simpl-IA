"""
Interactive YOLO annotator v2
--------------------------------
✓ Draw new bounding‑boxes  (Left‑click + drag)
✓ Select & MOVE a box      (Left‑click inside box  → drag)
✓ Select & RESIZE a box    (Left‑click near a corner  → drag)
✓ Delete a box             (Right‑click inside box)
✓ Save annotations         (Press 's')
✓ Quit                     (Press 'q' or 'ESC')

Output : labels.txt (YOLO format)
---------------------------------
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

Author : Jarvis‑style helper for Monsieur FLAGET
"""

import cv2
import os
from typing import List, Tuple

# === CONFIGURATION ===
IMAGE_PATH = "eceaa5eb-Fiche-aliment-Images-33-700x700.png"  # Chemin vers l'image
OUTPUT_TXT = "labels.txt"                                   # Fichier export
CLASS_ID = 0                                                # Classe par défaut (changer si besoin)
CORNER_TOL = 10                                             # px – sensibilité détection coins pour le redimensionnement

# === LOAD IMAGE ===
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image introuvable : {IMAGE_PATH}")

H, W = img.shape[:2]

# === STATE ===
boxes: List[Tuple[int, int, int, int]] = []  # [(x1,y1,x2,y2), ...]
clone = img.copy()
mode = "idle"          # idle | drawing | move | resize
selected_idx = -1      # index de la box sélectionnée
resize_corner = None   # 'tl' | 'tr' | 'bl' | 'br'
ix = iy = 0            # coordonnées initiales (pour dessin et déplacement)

def draw_boxes():
    """Redessine toutes les boxes sur l'image clone."""
    global clone
    clone = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = (0, 255, 0)
        thickness = 2
        if i == selected_idx:
            color = (0, 255, 255)  # highlight la box sélectionnée
            thickness = 3
        cv2.rectangle(clone, (x1, y1), (x2, y2), color, thickness)


def inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def near_corner(x, y, box):
    """Renvoie le coin près du point (None si éloigné)"""
    x1, y1, x2, y2 = box
    corners = {
        'tl': (x1, y1),
        'tr': (x2, y1),
        'bl': (x1, y2),
        'br': (x2, y2),
    }
    for name, (cx, cy) in corners.items():
        if abs(x - cx) < CORNER_TOL and abs(y - cy) < CORNER_TOL:
            return name
    return None


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))


def yolo_format(box):
    """(x1,y1,x2,y2) -> YOLO string"""
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / W
    y_center = ((y1 + y2) / 2) / H
    width = abs(x2 - x1) / W
    height = abs(y2 - y1) / H
    return f"{CLASS_ID} {x_center} {y_center} {width} {height}"

# === MOUSE CALLBACK ===

def mouse_cb(event, x, y, flags, param):
    global mode, ix, iy, boxes, selected_idx, resize_corner

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1) Cherche une box existante (en commençant par la plus haute)
        selected_idx = -1
        for i in reversed(range(len(boxes))):
            if inside_box(x, y, boxes[i]):
                selected_idx = i
                corner = near_corner(x, y, boxes[i])
                if corner:
                    mode = "resize"
                    resize_corner = corner
                else:
                    mode = "move"
                    ix, iy = x, y  # point de référence pour le déplacement
                break

        if selected_idx == -1:
            # 2) Pas de box → on commence à dessiner
            mode = "drawing"
            ix, iy = x, y
            boxes.append((x, y, x, y))  # placeholder
            selected_idx = len(boxes) - 1
        draw_boxes()

    elif event == cv2.EVENT_MOUSEMOVE:
        if mode == "drawing" and selected_idx != -1:
            x1, y1 = ix, iy
            x2, y2 = x, y
            boxes[selected_idx] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            draw_boxes()
        elif mode == "move" and selected_idx != -1:
            dx, dy = x - ix, y - iy
            x1, y1, x2, y2 = boxes[selected_idx]
            new_box = (
                clamp(x1 + dx, 0, W - 1),
                clamp(y1 + dy, 0, H - 1),
                clamp(x2 + dx, 0, W - 1),
                clamp(y2 + dy, 0, H - 1),
            )
            boxes[selected_idx] = new_box
            ix, iy = x, y
            draw_boxes()
        elif mode == "resize" and selected_idx != -1:
            x1, y1, x2, y2 = boxes[selected_idx]
            if resize_corner == 'tl':
                x1, y1 = x, y
            elif resize_corner == 'tr':
                x2, y1 = x, y
            elif resize_corner == 'bl':
                x1, y2 = x, y
            elif resize_corner == 'br':
                x2, y2 = x, y
            boxes[selected_idx] = (
                clamp(min(x1, x2), 0, W - 1),
                clamp(min(y1, y2), 0, H - 1),
                clamp(max(x1, x2), 0, W - 1),
                clamp(max(y1, y2), 0, H - 1),
            )
            draw_boxes()

    elif event == cv2.EVENT_LBUTTONUP:
        if mode in {"drawing", "move", "resize"}:
            mode = "idle"
            resize_corner = None
            draw_boxes()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Supprimer box si clic droit à l'intérieur
        for i, b in enumerate(boxes):
            if inside_box(x, y, b):
                del boxes[i]
                mode = "idle"
                selected_idx = -1
                draw_boxes()
                break

# === INIT ===
cv2.namedWindow("YOLO Annotator")
cv2.setMouseCallback("YOLO Annotator", mouse_cb)

print("[INFO] Contrôles :")
print(" - LMB + drag : créer une box")
print(" - LMB inside : déplacer | LMB near corner : redimensionner")
print(" - RMB inside : supprimer la box")
print(" - S : sauvegarde YOLO → labels.txt")
print(" - Q / ESC : quitter")

draw_boxes()

while True:
    cv2.imshow("YOLO Annotator", clone)
    key = cv2.waitKey(1) & 0xFF

    if key in {ord('q'), 27}:  # q ou ESC
        break
    elif key == ord('s'):
        with open(OUTPUT_TXT, "w") as f:
            for b in boxes:
                f.write(yolo_format(b) + "\n")
        print(f"[INFO] Annotations sauvegardées → {OUTPUT_TXT} ({len(boxes)} box{'es' if len(boxes)!=1 else ''})")

cv2.destroyAllWindows()

