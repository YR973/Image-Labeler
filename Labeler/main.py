import sys
import os
import json
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QAction
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QToolBar,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
    QMessageBox
)

# ---------------------- Label configuration ---------------------- #

@dataclass
class ClassLabel:
    id: int
    name: str
    color: QColor  # for overlay / preview


CLASS_MAP = {
    0: ClassLabel(0, "background", QColor(0, 0, 0, 0)),
    1: ClassLabel(1, "tooth", QColor(255, 0, 0, 120)),
    2: ClassLabel(2, "canal", QColor(0, 255, 0, 120)),
    3: ClassLabel(3, "caries", QColor(0, 0, 255, 120)),
    4: ClassLabel(4, "periapical lesion", QColor(255, 255, 0, 120)),
    5: ClassLabel(5, "gutta-percha", QColor(255, 0, 255, 120)),
}

VALID_IDS = {0, 1, 2, 3, 4, 5}


# ---------------------- Canvas widget ---------------------- #

class LabelCanvas(QWidget):

    def get_per_class_masks(self):
        """Return dict of class_id -> binary mask (uint8, 0 or 255) for all classes."""
        h, w = self.mask.shape
        out = {}
        for cid in sorted(CLASS_MAP.keys()):
            layer = np.zeros((h, w), dtype=np.uint8)
            layer[self.mask == cid] = 255
            out[cid] = layer
        return out

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(512, 512)

        self.image = None          # QImage 512x512
        self.gray_img = None       # numpy grayscale 512x512
        self.mask = None           # numpy uint8 512x512

        self.current_class_id = 1
        self.current_tool = "polygon"   # polygon, brush, wand, pan

        self.mask_opacity = 0.5

        # polygon
        self.current_polygon = []       # list[QPointF]
        self.polygons = []              # polygons for JSON

        # undo / redo (for mask)
        self.undo_stack = []
        self.redo_stack = []
        self._operation_active = False
        self._mask_before_operation = None

        # brush
        self.brush_radius = 6
        self.brush_drawing = False
        self.brush_erasing = False

        # pan / zoom
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.panning = False
        self._last_mouse_pos = None

        # magic wand
        self.wand_threshold = 12

    # ---------------------- Image loading ---------------------- #

    def load_image(self, path: str):
        """Load base image (and reset mask/polygons)."""
        pil_img = Image.open(path).convert("RGB")
        pil_img = pil_img.resize((512, 512), Image.BILINEAR)

        self.image = self.pil_to_qimage(pil_img)
        self.gray_img = np.array(pil_img.convert("L"), dtype=np.uint8)
        self.mask = np.zeros((512, 512), dtype=np.uint8)

        self.polygons = []
        self.current_polygon = []

        self.undo_stack.clear()
        self.redo_stack.clear()

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        self.update()

    def pil_to_qimage(self, img: Image.Image) -> QImage:
        arr = np.array(img)
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        return QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    # allow main window to inject an existing mask & polygons
    def set_mask_and_polygons(self, mask: np.ndarray | None, polygons: list | None):
        if mask is not None:
            if mask.shape != (512, 512):
                # resize if needed, but try to avoid this ideally
                mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            self.mask = mask.astype(np.uint8)
        else:
            self.mask = np.zeros((512, 512), dtype=np.uint8)

        self.polygons = polygons if polygons is not None else []
        self.current_polygon = []
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update()

    # ---------------------- Undo / redo ---------------------- #

    def push_undo(self):
        if self.mask is not None:
            self.undo_stack.append(self.mask.copy())
            self.redo_stack.clear()

    def undo(self):
        """Undo for mask; polygon points are handled in MainWindow."""
        if not self.undo_stack:
            return
        self.redo_stack.append(self.mask.copy())
        self.mask = self.undo_stack.pop()
        self.update()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self.mask.copy())
        self.mask = self.redo_stack.pop()
        self.update()

    # ---------------------- Painting ---------------------- #

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(25, 25, 25))

        if self.image is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Open an image…")
            return

        painter.setRenderHint(QPainter.Antialiasing, True)

        # pan / zoom
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.scale, self.scale)

        # base image
        painter.drawPixmap(0, 0, QPixmap.fromImage(self.image))

        # overlay mask
        if self.mask is not None and self.mask_opacity > 0:
            h, w = self.mask.shape
            overlay_np = np.zeros((h, w, 4), dtype=np.uint8)

            for cid, label in CLASS_MAP.items():
                if cid == 0:
                    continue
                mask_bits = (self.mask == cid)
                if not np.any(mask_bits):
                    continue

                col = QColor(label.color)
                col.setAlpha(int(col.alpha() * self.mask_opacity))

                overlay_np[mask_bits, 0] = col.red()
                overlay_np[mask_bits, 1] = col.green()
                overlay_np[mask_bits, 2] = col.blue()
                overlay_np[mask_bits, 3] = col.alpha()

            overlay = QImage(
                overlay_np.data,
                w, h,
                overlay_np.strides[0],
                QImage.Format_RGBA8888,
            )
            painter.drawImage(0, 0, overlay)

        # current polygon
        if self.current_polygon:
            pen = QPen(Qt.white)
            pen.setWidth(2)
            painter.setPen(pen)
            for i in range(len(self.current_polygon) - 1):
                painter.drawLine(self.current_polygon[i], self.current_polygon[i + 1])
            for p in self.current_polygon:
                painter.drawEllipse(p, 3, 3)

    # ---------------------- Tool helpers ---------------------- #

    def set_tool(self, tool: str):
        self.current_tool = tool

    def set_class_id(self, cid: int):
        if cid in VALID_IDS:
            self.current_class_id = cid

    def set_mask_opacity(self, value: int):
        self.mask_opacity = value / 100.0
        self.update()

    # ---------------------- Coordinate mapping ---------------------- #

    def screen_to_image(self, pos):
        x = (pos.x() - self.offset_x) / self.scale
        y = (pos.y() - self.offset_y) / self.scale
        return x, y

    # ---------------------- Mouse events ---------------------- #

    def mousePressEvent(self, event):
        if self.image is None:
            return

        pos = event.position().toPoint()

        # panning
        if self.current_tool == "pan" or event.button() == Qt.MiddleButton:
            self.panning = True
            self._last_mouse_pos = pos
            return

        x, y = self.screen_to_image(pos)
        if not (0 <= x < 512 and 0 <= y < 512):
            return

        # polygon: left click = add point, right click = close polygon
        if self.current_tool == "polygon":
            if event.button() == Qt.LeftButton:
                # add a vertex
                self.current_polygon.append(QPointF(x, y))
                self.update()
            elif event.button() == Qt.RightButton:
                # close polygon if enough points
                if len(self.current_polygon) >= 3:
                    self._operation_active = True
                    self._mask_before_operation = self.mask.copy()
                    self.commit_polygon()
                    self.finish_operation()
                # clear current polygon whether we committed or not
                self.current_polygon = []
                self.update()
            return

        # brush
        if self.current_tool == "brush" and event.button() in (Qt.LeftButton, Qt.RightButton):
            self._operation_active = True
            self._mask_before_operation = self.mask.copy()
            self.brush_drawing = True
            self.brush_erasing = (event.button() == Qt.RightButton)
            self.apply_brush(x, y)
            self.update()
            return

        # magic wand
        if self.current_tool == "wand" and event.button() == Qt.LeftButton:
            self._operation_active = True
            self._mask_before_operation = self.mask.copy()
            self.apply_wand(int(x), int(y))
            self.finish_operation()
            self.update()
            return

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()

        if self.panning and self._last_mouse_pos is not None:
            delta = pos - self._last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self._last_mouse_pos = pos
            self.update()
            return

        if self.current_tool == "brush" and self.brush_drawing:
            x, y = self.screen_to_image(pos)
            if 0 <= x < 512 and 0 <= y < 512:
                self.apply_brush(x, y)
                self.update()

    def mouseReleaseEvent(self, event):
        if self.current_tool == "pan":
            self.panning = False
            self._last_mouse_pos = None

        if self.current_tool == "brush" and self.brush_drawing:
            self.brush_drawing = False
            self.finish_operation()

    def mouseDoubleClickEvent(self, event):
        # no longer used for polygon closing
        pass

    def wheelEvent(self, event):
        if self.image is None:
            return

        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15

        self.scale = float(np.clip(self.scale * factor, 0.2, 5.0))

        cursor = event.position().toPoint()
        x_before, y_before = self.screen_to_image(cursor)
        x_after = (cursor.x() - self.offset_x) / self.scale
        y_after = (cursor.y() - self.offset_y) / self.scale

        self.offset_x += (x_after - x_before) * self.scale
        self.offset_y += (y_after - y_before) * self.scale

        self.update()

    # ---------------------- Drawing ops ---------------------- #

    def finish_operation(self):
        if self._operation_active and self._mask_before_operation is not None:
            self.undo_stack.append(self._mask_before_operation.copy())
            self.redo_stack.clear()
        self._operation_active = False
        self._mask_before_operation = None

    def apply_brush(self, x, y):
        if self.mask is None:
            return

        cx, cy = int(x), int(y)
        r = self.brush_radius

        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        circle = (xx * xx + yy * yy) <= r * r

        x_min = max(0, cx - r)
        x_max = min(511, cx + r)
        y_min = max(0, cy - r)
        y_max = min(511, cy + r)

        region = self.mask[y_min:y_max+1, x_min:x_max+1]
        sub = circle[
            (y_min - cy + r):(y_max - cy + r + 1),
            (x_min - cx + r):(x_max - cx + r + 1),
        ]

        if self.brush_erasing:
            region[sub] = 0
        else:
            region[sub] = self.current_class_id

    def commit_polygon(self):
        if self.mask is None or not self.current_polygon:
            return

        pts = np.array([[p.x(), p.y()] for p in self.current_polygon], dtype=np.int32)
        pts = np.clip(pts, 0, 511)
        cv2.fillPoly(self.mask, [pts.reshape((-1, 1, 2))], int(self.current_class_id))

        class_label = CLASS_MAP[self.current_class_id]
        self.polygons.append({
            "class_id": int(self.current_class_id),
            "label": class_label.name,
            "points": pts.tolist()
        })

    def apply_wand(self, x, y):
        if self.gray_img is None:
            return

        h, w = self.gray_img.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        seed_val = int(self.gray_img[y, x])
        tol = self.wand_threshold

        visited = np.zeros((h, w), dtype=bool)
        stack = [(x, y)]
        visited[y, x] = True

        while stack:
            cx, cy = stack.pop()
            val = int(self.gray_img[cy, cx])
            if abs(val - seed_val) <= tol:
                self.mask[cy, cx] = self.current_class_id
                for nx, ny in ((cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)):
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))

    # ---------------------- Export helpers ---------------------- #

    def get_mask_uint8(self):
        m = self.mask.copy().astype(np.uint8)
        m[~np.isin(m, list(VALID_IDS))] = 0
        return m

    def get_json(self, image_filename: str):
        return {
            "image": image_filename,
            "width": 512,
            "height": 512,
            "polygons": self.polygons,
        }

    def get_colored_mask(self):
        """Return RGB preview mask with class colors."""
        h, w = self.mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        color_map = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (255, 255, 0),
            5: (255, 0, 255),
        }

        for cid, col in color_map.items():
            rgb[self.mask == cid] = col

        return rgb


# ---------------------- Main window ---------------------- #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dental Mask Labeler (PySide6)")
        self.canvas = LabelCanvas(self)

        # Use the program/script directory, not the image directory
        self.base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.current_base_name = None

        # folder / image list for navigation
        self.image_folder = None
        self.image_files = []   # list of filenames in folder
        self.current_index = -1

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # toolbar
        tb = QToolBar()
        self.addToolBar(tb)

        act_open = QAction("Open Folder", self)
        act_open.triggered.connect(self.open_folder)
        tb.addAction(act_open)

        act_save = QAction("Save", self)
        act_save.triggered.connect(lambda: self.save_all(show_message=True))
        tb.addAction(act_save)

        act_undo = QAction("Undo", self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(self.handle_undo)
        tb.addAction(act_undo)

        act_redo = QAction("Redo", self)
        act_redo.setShortcut("Ctrl+Y")
        act_redo.triggered.connect(self.canvas.redo)
        tb.addAction(act_redo)

        # tool buttons
        tools_row = QHBoxLayout()
        layout.addLayout(tools_row)

        for name, tool in [
            ("Polygon (P)", "polygon"),
            ("Brush (B)", "brush"),
            ("Wand (M)", "wand"),
            ("Pan (Space)", "pan"),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, t=tool: self.canvas.set_tool(t))
            tools_row.addWidget(btn)

        # navigation buttons
        nav_row = QHBoxLayout()
        layout.addLayout(nav_row)

        btn_prev = QPushButton("Previous")
        btn_prev.clicked.connect(self.prev_image)
        nav_row.addWidget(btn_prev)

        btn_next = QPushButton("Next")
        btn_next.clicked.connect(self.next_image)
        nav_row.addWidget(btn_next)

        # class selector
        class_row = QHBoxLayout()
        layout.addLayout(class_row)

        class_row.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        for cid in sorted(CLASS_MAP.keys()):
            label = CLASS_MAP[cid]
            self.class_combo.addItem(f"{cid} - {label.name}", cid)
        self.class_combo.currentIndexChanged.connect(
            lambda i: self.canvas.set_class_id(self.class_combo.itemData(i))
        )
        class_row.addWidget(self.class_combo)

        # opacity slider
        op_row = QHBoxLayout()
        layout.addLayout(op_row)
        op_row.addWidget(QLabel("Mask opacity:"))
        op_slider = QSlider(Qt.Horizontal)
        op_slider.setRange(0, 100)
        op_slider.setValue(int(self.canvas.mask_opacity * 100))
        op_slider.valueChanged.connect(self.canvas.set_mask_opacity)
        op_row.addWidget(op_slider)

        # wand threshold slider
        wand_row = QHBoxLayout()
        layout.addLayout(wand_row)
        wand_row.addWidget(QLabel("Wand threshold:"))
        wand_slider = QSlider(Qt.Horizontal)
        wand_slider.setRange(1, 60)
        wand_slider.setValue(self.canvas.wand_threshold)
        wand_slider.valueChanged.connect(
            lambda v: setattr(self.canvas, "wand_threshold", v)
        )
        wand_row.addWidget(wand_slider)

        layout.addWidget(self.canvas)

        self.resize(900, 700)

    # shortcuts for tools / classes
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_P:
            self.canvas.set_tool("polygon")
        elif key == Qt.Key_B:
            self.canvas.set_tool("brush")
        elif key == Qt.Key_M:
            self.canvas.set_tool("wand")
        elif key == Qt.Key_Space:
            self.canvas.set_tool("pan")
        elif key == Qt.Key_Left:
            self.prev_image()
        elif key == Qt.Key_Right:
            self.next_image()
        elif Qt.Key_0 <= key <= Qt.Key_9:
            cid = key - Qt.Key_0
            if cid in VALID_IDS:
                self.canvas.set_class_id(cid)
                # sync combo
                for i in range(self.class_combo.count()):
                    if self.class_combo.itemData(i) == cid:
                        self.class_combo.setCurrentIndex(i)
                        break

    # ---------------------- Undo handler (including polygon points) ---------------------- #

    def handle_undo(self):
        # If currently drawing a polygon, undo removes the last point
        if self.canvas.current_tool == "polygon" and self.canvas.current_polygon:
            self.canvas.current_polygon.pop()
            self.canvas.update()
        else:
            self.canvas.undo()

    # ---------------------- Folder + image navigation ---------------------- #

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return

        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(exts)]
        files.sort()

        if not files:
            QMessageBox.warning(self, "No images", "No image files found in this folder.")
            return

        self.image_folder = folder
        self.image_files = files
        self.current_index = 0

        self.load_current_image()

    def load_current_image(self):
        if self.image_folder is None or self.current_index < 0 or self.current_index >= len(self.image_files):
            return

        filename = self.image_files[self.current_index]
        full_path = os.path.join(self.image_folder, filename)

        base = os.path.splitext(filename)[0]
        if base.endswith("_original"):
            base = base[:-9]
        self.current_base_name = base

        # load base image
        self.canvas.load_image(full_path)

        # try to load existing mask / polygons from previous session (autosave)
        images_dir = os.path.join(self.base_dir, "images_512")
        masks_dir = os.path.join(self.base_dir, "masks_512_cleaned")
        json_dir = os.path.join(self.base_dir, "labeled_data")

        mask = None
        polygons = None

        mask_path = os.path.join(masks_dir, f"{base}_mask.png")
        if os.path.exists(mask_path):
            try:
                m = np.array(Image.open(mask_path))
                if m.ndim == 2:
                    mask = m.astype(np.uint8)
            except Exception:
                mask = None

        json_path = os.path.join(json_dir, f"{base}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                polygons = data.get("polygons", [])
            except Exception:
                polygons = None

        self.canvas.set_mask_and_polygons(mask, polygons)

        self.setWindowTitle(
            f"Dental Mask Labeler – {base} ({self.current_index + 1}/{len(self.image_files)})"
        )

    def autosave_current(self):
        if self.canvas.image is not None and self.current_base_name is not None:
            self.save_all(show_message=False)

    def next_image(self):
        if not self.image_files:
            return
        # autosave current before moving
        self.autosave_current()
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
        else:
            QMessageBox.information(self, "End", "This is the last image.")

    def prev_image(self):
        if not self.image_files:
            return
        # autosave current before moving
        self.autosave_current()
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
        else:
            QMessageBox.information(self, "Start", "This is the first image.")

    # ---------------------- Saving (with overwrite & optional popup) ---------------------- #

    def save_all(self, show_message: bool = True):
        if self.canvas.image is None or self.current_base_name is None:
            if show_message:
                QMessageBox.warning(self, "Nothing to save", "Load an image first.")
            return

        base = self.current_base_name

        images_dir = os.path.join(self.base_dir, "images_512")
        masks_dir = os.path.join(self.base_dir, "masks_512_cleaned")
        json_dir = os.path.join(self.base_dir, "labeled_data")
        layers_dir = os.path.join(self.base_dir, "layers_512")  # per-class layers
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(layers_dir, exist_ok=True)

        # 1) original image 512x512 (saved as grayscale)
        img_filename = f"{base}_original.png"
        img_path = os.path.join(images_dir, img_filename)

        gray_np = np.array(self.canvas.gray_img, dtype=np.uint8)
        gray_img = Image.fromarray(gray_np, mode="L")
        gray_img.save(img_path)


# 2) true label mask (0–5, uint8, grayscale)
        mask_uint8 = self.canvas.get_mask_uint8()
        mask_img = Image.fromarray(mask_uint8, mode="L")
        mask_filename = f"{base}_mask.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        mask_img.save(mask_path)

        # 3) colored preview mask (RGB)
        preview_rgb = self.canvas.get_colored_mask()
        preview_img = Image.fromarray(preview_rgb, mode="RGB")
        preview_filename = f"{base}_mask_preview.png"
        preview_path = os.path.join(masks_dir, preview_filename)
        preview_img.save(preview_path)

        # 4) JSON metadata
        json_data = self.canvas.get_json(img_filename)
        json_filename = f"{base}.json"
        json_path = os.path.join(json_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # 5) per-class layer images (binary masks), saved even if empty
        per_class = self.canvas.get_per_class_masks()
        saved_layer_paths = []
        for cid, layer in per_class.items():
            label_name = CLASS_MAP[cid].name.replace(" ", "_")
            layer_filename = f"{base}_layer_{cid}_{label_name}.png"
            layer_path = os.path.join(layers_dir, layer_filename)
            Image.fromarray(layer, mode="L").save(layer_path)
            saved_layer_paths.append(layer_path)

        if show_message:
            QMessageBox.information(
                self,
                "Saved",
                f"Saved:\n{img_path}\n{mask_path}\n{preview_path}\n{json_path}\n" +
                "\n".join(saved_layer_paths),
                )


# ---------------------- Main entry ---------------------- #

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
