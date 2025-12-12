"""
TEED Edge Labeler - Qt5 GUI
Edge Map 라벨링 도구 for TEED Training
"""

import sys
import os
import json
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import uic

# UI 파일 경로
UI_FILE = os.path.join(os.path.dirname(__file__), 'ui', 'teed_labeler.ui')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI 로드
        uic.loadUi(UI_FILE, self)

        # 변수 초기화
        self.image_folder = ""
        self.image_files = []
        self.current_idx = 0
        self.img = None
        self.img_color = None

        # 그리기 관련
        self.lines = []  # [(pt1, pt2), ...]
        self.current_line_start = None
        self.freehand_points = []  # 자유 그리기 포인트
        self.is_drawing = False

        # Edge Map 저장용
        self.edge_maps = {}  # {image_path: edge_map_array}
        self.labeled_images = set()

        # 이벤트 연결
        self.setup_connections()

        # 마우스 이벤트 설정
        self.label_image.mousePressEvent = self.image_mouse_press
        self.label_image.mouseMoveEvent = self.image_mouse_move
        self.label_image.mouseReleaseEvent = self.image_mouse_release

        # 상태바 초기화
        self.statusbar.showMessage("Load an image folder to start labeling")

    def setup_connections(self):
        """버튼/메뉴 이벤트 연결"""
        self.btn_load_folder.clicked.connect(self.load_folder)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_save_current.clicked.connect(self.save_current_edge_map)
        self.btn_export.clicked.connect(self.export_dataset)
        self.comboBox_scale.currentIndexChanged.connect(self.update_output_size_label)

        # 기본값: 1/4 크기
        self.comboBox_scale.setCurrentIndex(2)

    def load_folder(self):
        """이미지 폴더 로드"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder",
            "D:/Image"
        )

        if not folder:
            return

        self.image_folder = folder
        self.image_files = []

        for f in os.listdir(folder):
            if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                self.image_files.append(os.path.join(folder, f))

        self.image_files.sort()

        if self.image_files:
            self.current_idx = 0
            self.load_image(0)
            self.label_folder_path.setText(os.path.basename(folder))
            self.label_total_count.setText(str(len(self.image_files)))
            self.statusbar.showMessage(f"Loaded {len(self.image_files)} images - Click to draw edges")
        else:
            QMessageBox.warning(self, "Warning", "No image files found.")

    def load_image(self, idx):
        """이미지 로드"""
        if 0 <= idx < len(self.image_files):
            self.current_idx = idx
            img_path = self.image_files[idx]

            self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            self.img_color = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

            # 기존 라인 데이터 복원
            if img_path in self.edge_maps:
                self.lines = self.edge_maps[img_path].get('lines', [])
            else:
                self.lines = []

            self.current_line_start = None
            self.freehand_points = []

            self.update_display()
            self.update_edge_preview()
            self.update_image_info()

    def update_image_info(self):
        """이미지 정보 업데이트"""
        if self.image_files:
            name = os.path.basename(self.image_files[self.current_idx])
            self.label_image_info.setText(f"{self.current_idx + 1} / {len(self.image_files)} : {name}")
            self.label_lines_count.setText(str(len(self.lines)))
            self.label_labeled_count.setText(str(len(self.labeled_images)))
            self.update_output_size_label()

    def update_output_size_label(self):
        """출력 크기 라벨 업데이트"""
        if self.img is None:
            return

        h, w = self.img.shape
        scale_idx = self.comboBox_scale.currentIndex()
        scale_map = {0: 1.0, 1: 0.5, 2: 0.25}
        scale = scale_map.get(scale_idx, 0.25)

        out_w = int(w * scale)
        out_h = int(h * scale)
        self.label_output_size.setText(f"Output: {w}x{h} -> {out_w}x{out_h}")

    def update_display(self):
        """화면 업데이트"""
        if self.img is None:
            return

        display = self.img_color.copy()

        # 저장된 라인 그리기
        for line in self.lines:
            if len(line) == 2:
                cv2.line(display, line[0], line[1], (0, 255, 0), self.spinBox_thickness.value())
            elif len(line) > 2:
                # freehand 라인
                pts = np.array(line, dtype=np.int32)
                cv2.polylines(display, [pts], False, (0, 255, 0), self.spinBox_thickness.value())

        # 현재 그리고 있는 라인 (시작점만 있을 때)
        if self.current_line_start:
            cv2.circle(display, self.current_line_start, 5, (0, 0, 255), -1)

        # Freehand 진행 중
        if self.freehand_points and len(self.freehand_points) > 1:
            pts = np.array(self.freehand_points, dtype=np.int32)
            cv2.polylines(display, [pts], False, (255, 255, 0), self.spinBox_thickness.value())

        # QPixmap으로 변환
        h, w, ch = display.shape
        bytes_per_line = ch * w
        q_img = QImage(display.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.label_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.label_image.setPixmap(scaled)
        self.update_image_info()

    def update_edge_preview(self):
        """Edge Map 미리보기 업데이트"""
        if self.img is None:
            return

        h, w = self.img.shape
        edge_map = np.zeros((h, w), dtype=np.uint8)

        # 라인 그리기
        thickness = self.spinBox_thickness.value()
        for line in self.lines:
            if len(line) == 2:
                cv2.line(edge_map, line[0], line[1], 255, thickness)
            elif len(line) > 2:
                pts = np.array(line, dtype=np.int32)
                cv2.polylines(edge_map, [pts], False, 255, thickness)

        # QPixmap으로 변환
        q_img = QImage(edge_map.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.label_edge_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_edge_preview.setPixmap(scaled)

    def get_image_coords(self, pos):
        """위젯 좌표 → 이미지 좌표 변환"""
        if self.img is None:
            return None

        label_w = self.label_image.width()
        label_h = self.label_image.height()
        img_h, img_w = self.img.shape

        scale = min(label_w / img_w, label_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2

        x = int((pos.x() - offset_x) / scale)
        y = int((pos.y() - offset_y) / scale)

        if 0 <= x < img_w and 0 <= y < img_h:
            return (x, y)
        return None

    def image_mouse_press(self, event):
        """마우스 클릭"""
        if self.img is None:
            return

        coords = self.get_image_coords(event.pos())
        if not coords:
            return

        if event.button() == Qt.LeftButton:
            if self.radio_line.isChecked():
                # Line 모드
                if self.current_line_start is None:
                    self.current_line_start = coords
                    self.statusbar.showMessage(f"Line start: {coords} - Click to set end point")
                else:
                    # 라인 완성
                    self.lines.append((self.current_line_start, coords))
                    self.statusbar.showMessage(f"Line added: {self.current_line_start} -> {coords}")
                    self.current_line_start = None
            else:
                # Freehand 모드
                self.is_drawing = True
                self.freehand_points = [coords]

            self.update_display()
            self.update_edge_preview()

        elif event.button() == Qt.RightButton:
            # 취소
            self.current_line_start = None
            self.freehand_points = []
            self.is_drawing = False
            self.update_display()

    def image_mouse_move(self, event):
        """마우스 이동"""
        coords = self.get_image_coords(event.pos())
        if coords:
            self.statusbar.showMessage(f"Position: {coords}")

            if self.is_drawing and self.radio_freehand.isChecked():
                self.freehand_points.append(coords)
                self.update_display()

    def image_mouse_release(self, event):
        """마우스 릴리즈"""
        if self.is_drawing and self.radio_freehand.isChecked():
            if len(self.freehand_points) > 1:
                self.lines.append(self.freehand_points.copy())
                self.statusbar.showMessage(f"Freehand line added: {len(self.freehand_points)} points")
            self.freehand_points = []
            self.is_drawing = False
            self.update_display()
            self.update_edge_preview()

    def clear_all(self):
        """모든 라인 지우기"""
        self.lines = []
        self.current_line_start = None
        self.freehand_points = []
        self.update_display()
        self.update_edge_preview()
        self.statusbar.showMessage("All lines cleared")

    def undo(self):
        """마지막 라인 취소"""
        if self.lines:
            self.lines.pop()
            self.update_display()
            self.update_edge_preview()
            self.statusbar.showMessage(f"Undo - {len(self.lines)} lines remaining")

    def save_current_edge_map(self):
        """현재 Edge Map 저장"""
        if self.img is None or not self.image_files:
            return

        if not self.lines:
            QMessageBox.warning(self, "Warning", "No edges drawn. Draw some edges first.")
            return

        img_path = self.image_files[self.current_idx]
        self.edge_maps[img_path] = {
            'lines': self.lines.copy(),
            'thickness': self.spinBox_thickness.value()
        }
        self.labeled_images.add(img_path)

        self.update_image_info()
        self.statusbar.showMessage(f"Edge map saved for {os.path.basename(img_path)}")

    def prev_image(self):
        """이전 이미지"""
        # 현재 이미지 자동 저장
        if self.lines:
            self.save_current_edge_map()

        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)

    def next_image(self):
        """다음 이미지"""
        # 현재 이미지 자동 저장
        if self.lines:
            self.save_current_edge_map()

        if self.current_idx < len(self.image_files) - 1:
            self.load_image(self.current_idx + 1)

    def export_dataset(self):
        """TEED 학습용 데이터셋 내보내기"""
        if not self.labeled_images:
            QMessageBox.warning(self, "Warning", "No labeled images. Label some images first.")
            return

        # 출력 폴더 선택
        output_folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            os.path.dirname(self.image_folder)
        )

        if not output_folder:
            return

        # 스케일 설정
        scale_idx = self.comboBox_scale.currentIndex()
        scale_map = {0: 1.0, 1: 0.5, 2: 0.25}
        scale = scale_map.get(scale_idx, 0.25)

        # 폴더 구조 생성
        imgs_folder = os.path.join(output_folder, 'imgs', 'train')
        edge_maps_folder = os.path.join(output_folder, 'edge_maps', 'train')
        os.makedirs(imgs_folder, exist_ok=True)
        os.makedirs(edge_maps_folder, exist_ok=True)

        # 학습 리스트 생성
        train_pairs = []

        for img_path in self.labeled_images:
            if img_path not in self.edge_maps:
                continue

            data = self.edge_maps[img_path]
            lines = data['lines']
            thickness = data.get('thickness', 2)

            # 원본 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # Edge Map 생성 (원본 크기)
            edge_map = np.zeros((h, w), dtype=np.uint8)
            for line in lines:
                if len(line) == 2:
                    cv2.line(edge_map, line[0], line[1], 255, thickness)
                elif len(line) > 2:
                    pts = np.array(line, dtype=np.int32)
                    cv2.polylines(edge_map, [pts], False, 255, thickness)

            # 스케일 적용
            if scale != 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                edge_map = cv2.resize(edge_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Edge map 이진화 (리사이즈 후 다시 0/255로)
                _, edge_map = cv2.threshold(edge_map, 127, 255, cv2.THRESH_BINARY)

            # 파일명
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img_filename = f"{base_name}.png"
            edge_filename = f"{base_name}.png"

            # 이미지 저장
            cv2.imwrite(os.path.join(imgs_folder, img_filename), img)
            cv2.imwrite(os.path.join(edge_maps_folder, edge_filename), edge_map)

            # 학습 리스트에 추가
            train_pairs.append([
                f"imgs/train/{img_filename}",
                f"edge_maps/train/{edge_filename}"
            ])

        # train_pair.lst (JSON 형식) 저장
        train_list_path = os.path.join(output_folder, 'train_pair.lst')
        with open(train_list_path, 'w') as f:
            json.dump(train_pairs, f, indent=2)

        # test_pair.lst도 동일하게 생성 (검증용)
        test_list_path = os.path.join(output_folder, 'test_pair.lst')
        with open(test_list_path, 'w') as f:
            json.dump(train_pairs, f, indent=2)

        # 출력 크기 계산
        if train_pairs:
            sample_img = cv2.imread(os.path.join(imgs_folder, os.path.basename(train_pairs[0][0])))
            if sample_img is not None:
                out_h, out_w = sample_img.shape[:2]
                size_info = f"Output size: {out_w}x{out_h}"
            else:
                size_info = ""
        else:
            size_info = ""

        QMessageBox.information(
            self, "Export Complete",
            f"Dataset exported successfully!\n\n"
            f"Location: {output_folder}\n"
            f"Images: {len(train_pairs)}\n"
            f"Scale: {scale}\n"
            f"{size_info}\n\n"
            f"Files created:\n"
            f"- imgs/train/\n"
            f"- edge_maps/train/\n"
            f"- train_pair.lst\n"
            f"- test_pair.lst"
        )

        self.statusbar.showMessage(f"Exported {len(train_pairs)} images to {output_folder}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
