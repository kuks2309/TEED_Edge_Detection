"""
TEED Edge Labeler & Trainer - Qt5 GUI
Edge Map 라벨링, 학습, 테스트 통합 도구
"""

import sys
import os
import json
import cv2
import numpy as np
import time
import configparser
from datetime import datetime

# PyTorch를 PyQt5보다 먼저 import (DLL 충돌 방지)
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt5 import uic

# 설정 파일 경로
CONFIG_FILE = 'D:/FITO_2026/Prevision/config/teed_settings.ini'

# UI 파일 경로
UI_FILE = os.path.join(os.path.dirname(__file__), 'ui', 'teed_labeler.ui')


class TrainingThread(QThread):
    """학습 스레드"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, float, float, int)  # epoch, total, loss, best_loss, patience
    finished_signal = pyqtSignal(str)  # history_dir

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            from teed_model import TED
            from teed_dataset import BipedDataset, TestDataset, dataset_info
            from teed_loss import cats_loss, bdcn_loss2
            from torch.utils.data import DataLoader
            import torch.optim as optim

            # Config
            data_dir = self.config['data_dir']
            epochs = self.config['epochs']
            batch_size = self.config['batch_size']
            lr = self.config['lr']
            patience = self.config['patience']
            output_name = self.config['output_name']
            pretrained_path = self.config.get('pretrained_path', '')
            use_pretrained = self.config.get('use_pretrained', True)

            # Device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_signal.emit(f"Device: {device}")

            # Output directory
            base_dir = os.path.join('weights', output_name)
            train_time = datetime.now().strftime('%Y%m%d_%H%M')
            history_dir = os.path.join(base_dir, 'history', train_time)
            deploy_dir = os.path.join(base_dir, 'deploy')
            os.makedirs(history_dir, exist_ok=True)
            os.makedirs(deploy_dir, exist_ok=True)

            self.log_signal.emit(f"History dir: {history_dir}")
            self.log_signal.emit(f"Deploy dir: {deploy_dir}")

            # Model
            model = TED().to(device)

            # Load pretrained
            if use_pretrained and pretrained_path and os.path.exists(pretrained_path):
                model.load_state_dict(torch.load(pretrained_path, map_location=device))
                self.log_signal.emit(f"Loaded pretrained: {pretrained_path}")

            # Dataset
            train_inf = dataset_info('PREVISION', is_linux=False)
            train_inf['data_dir'] = data_dir

            class Args:
                mean_train = train_inf['mean'][:3]
                mean_test = train_inf['mean'][:3]
                train_data = 'PREVISION'
                train_list = 'train_pair.lst'
                test_list = 'test_pair.lst'
                test_img_width = train_inf['img_width']
                test_img_height = train_inf['img_height']
                up_scale = False

            args = Args()

            dataset_train = BipedDataset(
                data_dir,
                img_width=300,
                img_height=300,
                train_mode='train',
                arg=args
            )

            dataloader_train = DataLoader(
                dataset_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )

            self.log_signal.emit(f"Train samples: {len(dataset_train)}")

            # Loss & Optimizer
            criterion1 = cats_loss
            criterion2 = bdcn_loss2
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)

            l_weight0 = [1.1, 0.7, 1.1, 1.3]
            l_weight = [[0.05, 2.], [0.05, 2.], [0.01, 1.], [0.01, 3.]]

            # Training loop
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                if not self.is_running:
                    self.log_signal.emit("Training stopped by user")
                    break

                model.train()
                loss_avg = []

                for batch_id, sample_batched in enumerate(dataloader_train):
                    if not self.is_running:
                        break

                    images = sample_batched['images'].to(device)
                    labels = sample_batched['labels'].to(device)

                    preds_list = model(images)
                    loss1 = sum([criterion2(preds, labels, l_w) for preds, l_w in zip(preds_list[:-1], l_weight0)])
                    loss2 = criterion1(preds_list[-1], labels, l_weight[-1], device)
                    tLoss = loss2 + loss1

                    optimizer.zero_grad()
                    tLoss.backward()
                    optimizer.step()
                    loss_avg.append(tLoss.item())

                avg_loss = np.array(loss_avg).mean()
                self.log_signal.emit(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

                # Save best & Early Stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_path = os.path.join(history_dir, 'best_model.pth')
                    torch.save(model.state_dict(), best_path)
                    self.log_signal.emit(f"Saved best model (Loss: {best_loss:.4f})")
                else:
                    patience_counter += 1
                    self.log_signal.emit(f"No improvement. Patience: {patience_counter}/{patience}")

                self.progress_signal.emit(epoch + 1, epochs, avg_loss, best_loss, patience_counter)

                if patience_counter >= patience:
                    self.log_signal.emit(f"Early stopping at epoch {epoch + 1}!")
                    break

                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    ckpt_path = os.path.join(history_dir, f'{epoch + 1}_model.pth')
                    torch.save(model.state_dict(), ckpt_path)

            # Save train info
            train_info_path = os.path.join(history_dir, 'train_info.txt')
            with open(train_info_path, 'w', encoding='utf-8') as f:
                f.write(f"Training Time: {train_time}\n")
                f.write(f"Total Epochs: {epoch + 1}\n")
                f.write(f"Best Loss: {best_loss:.4f}\n")
                f.write(f"Train Samples: {len(dataset_train)}\n")
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"Learning Rate: {lr}\n")
                f.write(f"Early Stopping Patience: {patience}\n")
                f.write(f"Device: {device}\n")

            self.log_signal.emit("=" * 50)
            self.log_signal.emit("Training Complete!")
            self.log_signal.emit(f"Best Loss: {best_loss:.4f}")
            self.finished_signal.emit(history_dir)

        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}")
            import traceback
            self.log_signal.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI 로드
        uic.loadUi(UI_FILE, self)

        # ===== Labeling 탭 변수 =====
        self.image_folder = ""
        self.image_files = []
        self.current_idx = 0
        self.img = None
        self.img_color = None
        self.lines = []
        self.current_line_start = None
        self.freehand_points = []
        self.is_drawing = False
        self.edge_maps = {}
        self.labeled_images = set()

        # ===== Training 탭 변수 =====
        self.training_thread = None
        self.dataset_path = ""
        self.pretrained_model_path = "weights/Prevision_MLCC/BIPED/5/5_model.pth"

        # ===== Test 탭 변수 =====
        self.test_model = None
        self.test_model_path = "weights/Prevision_MLCC/deploy/best_model.pth"
        self.test_images = []
        self.test_current_idx = 0
        self.test_results = {}  # {img_path: {'edge_map': ..., 'h_lines': ..., 'v_lines': ...}}

        # 설정 로드
        self.load_settings()

        # 이벤트 연결
        self.setup_connections()

        # 마우스 이벤트 설정
        self.label_image.mousePressEvent = self.image_mouse_press
        self.label_image.mouseMoveEvent = self.image_mouse_move
        self.label_image.mouseReleaseEvent = self.image_mouse_release

        # 상태바 초기화
        self.statusbar.showMessage("TEED Edge Labeler & Trainer Ready")

    def load_settings(self):
        """설정 파일 로드"""
        config = configparser.ConfigParser()
        if os.path.exists(CONFIG_FILE):
            config.read(CONFIG_FILE)

            # Paths
            self.image_folder = config.get('Paths', 'ImageFolder', fallback='')
            self.test_model_path = config.get('Paths', 'ModelPath', fallback='weights/Prevision_MLCC/deploy/best_model.pth')
            self.dataset_path = config.get('Paths', 'TrainingDataDir', fallback='')

            # Training (UI에 설정)
            self.spinBox_epochs.setValue(config.getint('Training', 'Epochs', fallback=200))
            self.spinBox_batch_size.setValue(config.getint('Training', 'BatchSize', fallback=8))
            self.doubleSpinBox_lr.setValue(config.getfloat('Training', 'LearningRate', fallback=0.0001))
            self.spinBox_patience.setValue(config.getint('Training', 'Patience', fallback=20))

            # Test (UI에 설정)
            self.spinBox_threshold.setValue(config.getint('Test', 'BinaryThreshold', fallback=50))
            self.spinBox_min_length.setValue(config.getint('Test', 'MinLineLength', fallback=30))
            self.spinBox_max_gap.setValue(config.getint('Test', 'MaxLineGap', fallback=10))
            self.spinBox_angle_tolerance.setValue(config.getint('Test', 'AngleTolerance', fallback=30))

    def save_settings(self):
        """설정 파일 저장"""
        config = configparser.ConfigParser()

        # Paths
        config['Paths'] = {
            'ImageFolder': self.image_folder,
            'ModelPath': self.test_model_path,
            'TrainingDataDir': self.dataset_path
        }

        # Training
        config['Training'] = {
            'Epochs': str(self.spinBox_epochs.value()),
            'BatchSize': str(self.spinBox_batch_size.value()),
            'LearningRate': str(self.doubleSpinBox_lr.value()),
            'Patience': str(self.spinBox_patience.value())
        }

        # Test
        config['Test'] = {
            'BinaryThreshold': str(self.spinBox_threshold.value()),
            'MinLineLength': str(self.spinBox_min_length.value()),
            'MaxLineGap': str(self.spinBox_max_gap.value()),
            'AngleTolerance': str(self.spinBox_angle_tolerance.value())
        }

        with open(CONFIG_FILE, 'w') as f:
            config.write(f)

    def setup_connections(self):
        """버튼/메뉴 이벤트 연결"""
        # ===== Labeling 탭 =====
        self.btn_load_folder.clicked.connect(self.load_folder)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_save_current.clicked.connect(self.save_current_edge_map)
        self.btn_export.clicked.connect(self.export_dataset)
        self.comboBox_scale.currentIndexChanged.connect(self.update_output_size_label)
        self.comboBox_scale.setCurrentIndex(2)

        # ===== Training 탭 =====
        self.btn_select_dataset.clicked.connect(self.select_dataset)
        self.btn_select_pretrained.clicked.connect(self.select_pretrained)
        self.btn_start_training.clicked.connect(self.start_training)
        self.btn_stop_training.clicked.connect(self.stop_training)
        self.btn_deploy_model.clicked.connect(self.deploy_model)
        self.lineEdit_output_name.textChanged.connect(self.update_output_dir_label)

        # ===== Test 탭 =====
        self.btn_select_test_model.clicked.connect(self.select_test_model)
        self.btn_load_model.clicked.connect(self.load_test_model)
        self.btn_load_test_image.clicked.connect(self.load_test_image)
        self.btn_load_test_folder.clicked.connect(self.load_test_folder)
        self.btn_run_test.clicked.connect(self.run_test)
        self.btn_test_prev.clicked.connect(self.test_prev_image)
        self.btn_test_next.clicked.connect(self.test_next_image)
        self.btn_save_current_result.clicked.connect(self.save_current_result)
        self.btn_batch_save.clicked.connect(self.batch_save_results)
        self.btn_show_final_lines.clicked.connect(self.show_final_lines)

        # View mode radio buttons
        self.radio_view_original.toggled.connect(self.update_test_display)
        self.radio_view_edge.toggled.connect(self.update_test_display)
        self.radio_view_lines.toggled.connect(self.update_test_display)
        self.radio_view_overlay.toggled.connect(self.update_test_display)

    # ==================== Labeling 탭 기능 ====================

    def load_folder(self):
        """이미지 폴더 로드"""
        default_dir = self.image_folder if self.image_folder else "D:/Image"
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", default_dir
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
            self.statusbar.showMessage(f"Loaded {len(self.image_files)} images")
            self.save_settings()
        else:
            QMessageBox.warning(self, "Warning", "No image files found.")

    def load_image(self, idx):
        """이미지 로드"""
        if 0 <= idx < len(self.image_files):
            self.current_idx = idx
            img_path = self.image_files[idx]

            self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            self.img_color = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

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

        for line in self.lines:
            if len(line) == 2:
                cv2.line(display, line[0], line[1], (0, 255, 0), self.spinBox_thickness.value())
            elif len(line) > 2:
                pts = np.array(line, dtype=np.int32)
                cv2.polylines(display, [pts], False, (0, 255, 0), self.spinBox_thickness.value())

        if self.current_line_start:
            cv2.circle(display, self.current_line_start, 5, (0, 0, 255), -1)

        if self.freehand_points and len(self.freehand_points) > 1:
            pts = np.array(self.freehand_points, dtype=np.int32)
            cv2.polylines(display, [pts], False, (255, 255, 0), self.spinBox_thickness.value())

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

        thickness = self.spinBox_thickness.value()
        for line in self.lines:
            if len(line) == 2:
                cv2.line(edge_map, line[0], line[1], 255, thickness)
            elif len(line) > 2:
                pts = np.array(line, dtype=np.int32)
                cv2.polylines(edge_map, [pts], False, 255, thickness)

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
                if self.current_line_start is None:
                    self.current_line_start = coords
                else:
                    self.lines.append((self.current_line_start, coords))
                    self.current_line_start = None
            else:
                self.is_drawing = True
                self.freehand_points = [coords]

            self.update_display()
            self.update_edge_preview()

        elif event.button() == Qt.RightButton:
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

    def undo(self):
        """마지막 라인 취소"""
        if self.lines:
            self.lines.pop()
            self.update_display()
            self.update_edge_preview()

    def save_current_edge_map(self):
        """현재 Edge Map 저장"""
        if self.img is None or not self.image_files:
            return

        if not self.lines:
            QMessageBox.warning(self, "Warning", "No edges drawn.")
            return

        img_path = self.image_files[self.current_idx]
        self.edge_maps[img_path] = {
            'lines': self.lines.copy(),
            'thickness': self.spinBox_thickness.value()
        }
        self.labeled_images.add(img_path)
        self.update_image_info()

    def prev_image(self):
        """이전 이미지"""
        if self.lines:
            self.save_current_edge_map()
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)

    def next_image(self):
        """다음 이미지"""
        if self.lines:
            self.save_current_edge_map()
        if self.current_idx < len(self.image_files) - 1:
            self.load_image(self.current_idx + 1)

    def export_dataset(self):
        """TEED 학습용 데이터셋 내보내기 (날짜별 폴더 자동 생성)"""
        if not self.labeled_images:
            QMessageBox.warning(self, "Warning", "No labeled images.")
            return

        # 기본 training_data 폴더 경로
        default_training_dir = os.path.join(os.path.dirname(__file__), 'training_data')
        os.makedirs(default_training_dir, exist_ok=True)

        # 날짜별 폴더명 생성 (YYYYMMDD)
        date_folder = datetime.now().strftime('%Y%m%d')
        output_folder = os.path.join(default_training_dir, date_folder)

        # 폴더가 이미 존재하면 확인
        if os.path.exists(output_folder):
            reply = QMessageBox.question(
                self, "Folder Exists",
                f"Folder '{date_folder}' already exists.\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                # 사용자가 직접 폴더 선택
                output_folder = QFileDialog.getExistingDirectory(
                    self, "Select Output Folder", default_training_dir
                )
                if not output_folder:
                    return

        scale_idx = self.comboBox_scale.currentIndex()
        scale_map = {0: 1.0, 1: 0.5, 2: 0.25}
        scale = scale_map.get(scale_idx, 0.25)

        imgs_folder = os.path.join(output_folder, 'imgs', 'train')
        edge_maps_folder = os.path.join(output_folder, 'edge_maps', 'train')
        os.makedirs(imgs_folder, exist_ok=True)
        os.makedirs(edge_maps_folder, exist_ok=True)

        train_pairs = []

        for img_path in self.labeled_images:
            if img_path not in self.edge_maps:
                continue

            data = self.edge_maps[img_path]
            lines = data['lines']
            thickness = data.get('thickness', 2)

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            edge_map = np.zeros((h, w), dtype=np.uint8)
            for line in lines:
                if len(line) == 2:
                    cv2.line(edge_map, line[0], line[1], 255, thickness)
                elif len(line) > 2:
                    pts = np.array(line, dtype=np.int32)
                    cv2.polylines(edge_map, [pts], False, 255, thickness)

            if scale != 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                edge_map = cv2.resize(edge_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
                _, edge_map = cv2.threshold(edge_map, 127, 255, cv2.THRESH_BINARY)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img_filename = f"{base_name}.png"
            edge_filename = f"{base_name}.png"

            cv2.imwrite(os.path.join(imgs_folder, img_filename), img)
            cv2.imwrite(os.path.join(edge_maps_folder, edge_filename), edge_map)

            train_pairs.append([
                f"imgs/train/{img_filename}",
                f"edge_maps/train/{edge_filename}"
            ])

        train_list_path = os.path.join(output_folder, 'train_pair.lst')
        with open(train_list_path, 'w') as f:
            json.dump(train_pairs, f, indent=2)

        test_list_path = os.path.join(output_folder, 'test_pair.lst')
        with open(test_list_path, 'w') as f:
            json.dump(train_pairs, f, indent=2)

        QMessageBox.information(
            self, "Export Complete",
            f"Dataset exported!\n\nLocation: {output_folder}\nImages: {len(train_pairs)}"
        )

    # ==================== Training 탭 기능 ====================

    def select_dataset(self):
        """데이터셋 폴더 선택"""
        default_dir = self.dataset_path if self.dataset_path else "D:/FITO_2026/TEED/training_data"
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", default_dir
        )
        if folder:
            self.dataset_path = folder
            self.label_dataset_path.setText(folder)

            # train_pair.lst 확인
            train_list = os.path.join(folder, 'train_pair.lst')
            if os.path.exists(train_list):
                with open(train_list, 'r') as f:
                    pairs = json.load(f)
                self.label_train_samples.setText(str(len(pairs)))
            else:
                self.label_train_samples.setText("train_pair.lst not found")

            self.save_settings()

    def select_pretrained(self):
        """Pretrained 모델 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Pretrained Model",
            "weights/", "Model Files (*.pth)"
        )
        if file_path:
            self.pretrained_model_path = file_path
            self.label_pretrained_path.setText(os.path.basename(file_path))

    def update_output_dir_label(self):
        """출력 디렉토리 라벨 업데이트"""
        name = self.lineEdit_output_name.text()
        self.label_output_dir.setText(f"Output: weights/{name}/history/")

    def start_training(self):
        """학습 시작"""
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Select dataset folder first.")
            return

        # Config
        config = {
            'data_dir': self.dataset_path,
            'epochs': self.spinBox_epochs.value(),
            'batch_size': self.spinBox_batch_size.value(),
            'lr': self.doubleSpinBox_lr.value(),
            'patience': self.spinBox_patience.value(),
            'output_name': self.lineEdit_output_name.text(),
            'pretrained_path': self.pretrained_model_path if self.checkBox_use_pretrained.isChecked() else '',
            'use_pretrained': self.checkBox_use_pretrained.isChecked()
        }

        # UI 상태 변경
        self.btn_start_training.setEnabled(False)
        self.btn_stop_training.setEnabled(True)
        self.textEdit_log.clear()

        # 학습 스레드 시작
        self.training_thread = TrainingThread(config)
        self.training_thread.log_signal.connect(self.append_log)
        self.training_thread.progress_signal.connect(self.update_training_progress)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

    def stop_training(self):
        """학습 중지"""
        if self.training_thread:
            self.training_thread.stop()
            self.append_log("Stopping training...")

    def append_log(self, text):
        """로그 추가"""
        self.textEdit_log.append(text)
        self.textEdit_log.verticalScrollBar().setValue(
            self.textEdit_log.verticalScrollBar().maximum()
        )

    def update_training_progress(self, epoch, total, loss, best_loss, patience_count):
        """학습 진행 상황 업데이트"""
        progress = int(epoch / total * 100)
        self.progressBar_training.setValue(progress)
        self.label_current_epoch.setText(f"{epoch} / {total}")
        self.label_current_loss.setText(f"{loss:.4f}")
        self.label_best_loss.setText(f"{best_loss:.4f}")
        self.label_patience_count.setText(f"{patience_count} / {self.spinBox_patience.value()}")

    def training_finished(self, history_dir):
        """학습 완료"""
        self.btn_start_training.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.history_dir = history_dir

        QMessageBox.information(
            self, "Training Complete",
            f"Training finished!\n\nCheckpoints saved to:\n{history_dir}"
        )

    def deploy_model(self):
        """Best 모델을 deploy 폴더로 복사"""
        if not hasattr(self, 'history_dir') or not self.history_dir:
            # 최신 history 폴더 찾기
            output_name = self.lineEdit_output_name.text()
            base_dir = os.path.join('weights', output_name, 'history')
            if os.path.exists(base_dir):
                folders = sorted(os.listdir(base_dir), reverse=True)
                if folders:
                    self.history_dir = os.path.join(base_dir, folders[0])

        if not hasattr(self, 'history_dir') or not self.history_dir:
            QMessageBox.warning(self, "Warning", "No training history found.")
            return

        best_model = os.path.join(self.history_dir, 'best_model.pth')
        if not os.path.exists(best_model):
            QMessageBox.warning(self, "Warning", f"best_model.pth not found in {self.history_dir}")
            return

        output_name = self.lineEdit_output_name.text()
        deploy_dir = os.path.join('weights', output_name, 'deploy')
        os.makedirs(deploy_dir, exist_ok=True)

        import shutil
        deploy_path = os.path.join(deploy_dir, 'best_model.pth')
        shutil.copy(best_model, deploy_path)

        QMessageBox.information(
            self, "Deploy Complete",
            f"Model deployed to:\n{deploy_path}"
        )

    # ==================== Test 탭 기능 ====================

    def select_test_model(self):
        """테스트 모델 선택"""
        default_dir = os.path.dirname(self.test_model_path) if self.test_model_path else "weights/"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model",
            default_dir, "Model Files (*.pth)"
        )
        if file_path:
            self.test_model_path = file_path
            self.label_test_model_path.setText(file_path)
            self.save_settings()

    def load_test_model(self):
        """테스트 모델 로드"""
        try:
            from teed_model import TED

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.test_model = TED().to(device)
            self.test_model.load_state_dict(torch.load(self.test_model_path, map_location=device))
            self.test_model.eval()
            self.test_device = device

            self.label_model_status.setText(f"Model loaded ({device})")
            self.label_model_status.setStyleSheet("color: rgb(100, 255, 100);")
            self.statusbar.showMessage(f"Model loaded: {self.test_model_path}")

        except Exception as e:
            self.label_model_status.setText(f"Error: {str(e)}")
            self.label_model_status.setStyleSheet("color: rgb(255, 100, 100);")

    def load_test_image(self):
        """단일 테스트 이미지 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Image",
            "D:/Image", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.test_images = [file_path]
            self.test_current_idx = 0
            self.test_results = {}
            self.label_test_input_path.setText(os.path.basename(file_path))
            self.update_test_image_info()

    def load_test_folder(self):
        """테스트 이미지 폴더 로드"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Test Image Folder", "D:/Image"
        )
        if folder:
            self.test_images = []
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.test_images.append(os.path.join(folder, f))
            self.test_images.sort()
            self.test_current_idx = 0
            self.test_results = {}
            self.label_test_input_path.setText(f"{folder} ({len(self.test_images)} images)")
            self.update_test_image_info()

    def update_test_image_info(self):
        """테스트 이미지 정보 업데이트"""
        if self.test_images:
            self.label_test_image_info.setText(f"{self.test_current_idx + 1} / {len(self.test_images)}")
        else:
            self.label_test_image_info.setText("0 / 0")

    def run_test(self):
        """테스트 실행"""
        if self.test_model is None:
            QMessageBox.warning(self, "Warning", "Load model first.")
            return

        if not self.test_images:
            QMessageBox.warning(self, "Warning", "Load test images first.")
            return

        img_path = self.test_images[self.test_current_idx]
        img = cv2.imread(img_path)
        if img is None:
            return

        h_orig, w_orig = img.shape[:2]

        # 전처리: crop 크기 (resize 후 8배수 보장)
        w_crop = (w_orig // 4 // 8) * 8 * 4
        h_crop = (h_orig // 4 // 8) * 8 * 4

        # 중앙 crop
        y_start = (h_orig - h_crop) // 2
        x_start = (w_orig - w_crop) // 2
        img_proc = img[y_start:y_start+h_crop, x_start:x_start+w_crop]

        # 1/4 resize
        img_proc = cv2.resize(img_proc, (w_crop // 4, h_crop // 4), interpolation=cv2.INTER_AREA)

        mean = [104.007, 116.669, 122.679]
        img_tensor = img_proc.astype(np.float32)
        img_tensor -= mean
        img_tensor = img_tensor.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(self.test_device)

        # 추론
        t_start = time.perf_counter()
        with torch.no_grad():
            preds = self.test_model(img_tensor, single_test=True)
            pred = torch.sigmoid(preds[-1]).cpu().numpy()[0, 0]
        t_end = time.perf_counter()
        inference_time = (t_end - t_start) * 1000

        # 정규화
        pred_min = pred.min()
        pred_max = pred.max()
        if pred_max > pred_min:
            pred = (pred - pred_min) / (pred_max - pred_min)
        edge_map = (pred * 255).astype(np.uint8)

        # 직선 검출
        threshold = self.spinBox_threshold.value()
        min_length = self.spinBox_min_length.value()
        max_gap = self.spinBox_max_gap.value()
        angle_tol = self.spinBox_angle_tolerance.value()

        h_lines, v_lines, h_clusters, v_clusters = self.extract_lines(
            edge_map, threshold, min_length, max_gap, angle_tol
        )

        # 결과 저장
        self.test_results[img_path] = {
            'original': img_proc,
            'edge_map': edge_map,
            'h_lines': h_lines,
            'v_lines': v_lines,
            'h_clusters': h_clusters,
            'v_clusters': v_clusters,
            'inference_time': inference_time
        }

        # UI 업데이트
        self.label_h_lines_count.setText(f"{len(h_clusters)} clusters ({len(h_lines)} lines)")
        self.label_v_lines_count.setText(f"{len(v_clusters)} clusters ({len(v_lines)} lines)")
        self.label_inference_time.setText(f"{inference_time:.1f} ms")

        self.update_test_display()

    def extract_lines(self, edge_map, threshold=50, min_length=30, max_gap=10, angle_tolerance=30):
        """직선 추출"""
        _, binary = cv2.threshold(edge_map, threshold, 255, cv2.THRESH_BINARY)

        hough_lines = cv2.HoughLinesP(
            binary, rho=1, theta=np.pi/180, threshold=30,
            minLineLength=min_length, maxLineGap=max_gap
        )

        h_lines = []
        v_lines = []

        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi

                if -angle_tolerance <= angle <= angle_tolerance:
                    if x2 != x1:
                        a = (y2 - y1) / (x2 - x1)
                        b = y1 - a * x1
                    else:
                        a = 0
                        b = (y1 + y2) / 2
                    h_lines.append((x1, y1, x2, y2, length, angle, a, b))

                elif (90 - angle_tolerance) <= abs(angle) <= (90 + angle_tolerance):
                    if y2 != y1:
                        c = (x2 - x1) / (y2 - y1)
                        d = x1 - c * y1
                    else:
                        c = 0
                        d = (x1 + x2) / 2
                    v_lines.append((x1, y1, x2, y2, length, angle, c, d))

        h_lines.sort(key=lambda x: x[4], reverse=True)
        v_lines.sort(key=lambda x: x[4], reverse=True)

        h_clusters = self.cluster_lines(h_lines, threshold=10)
        v_clusters = self.cluster_lines(v_lines, threshold=10)

        return h_lines, v_lines, h_clusters, v_clusters

    def cluster_lines(self, lines, threshold=10):
        """직선 클러스터링"""
        if not lines:
            return []

        clusters = []
        for line in lines:
            x1, y1, x2, y2, length, angle, coef1, coef2 = line
            merged = False

            for cluster in clusters:
                if abs(cluster['coef2'] - coef2) < threshold:
                    cluster['lines'].append(line)
                    cluster['total_length'] += length
                    total_len = cluster['total_length']
                    cluster['coef1'] = (cluster['coef1'] * (total_len - length) + coef1 * length) / total_len
                    cluster['coef2'] = (cluster['coef2'] * (total_len - length) + coef2 * length) / total_len
                    merged = True
                    break

            if not merged:
                clusters.append({
                    'lines': [line],
                    'coef1': coef1,
                    'coef2': coef2,
                    'total_length': length
                })

        clusters.sort(key=lambda x: x['total_length'], reverse=True)
        return clusters

    def update_test_display(self):
        """테스트 결과 표시 업데이트"""
        if not self.test_images:
            return

        img_path = self.test_images[self.test_current_idx]
        if img_path not in self.test_results:
            # 아직 테스트 안 됨 - 원본 표시
            img = cv2.imread(img_path)
            if img is None:
                return
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            result = self.test_results[img_path]
            original = result['original']
            edge_map = result['edge_map']
            h_clusters = result['h_clusters']
            v_clusters = result['v_clusters']

            if self.radio_view_original.isChecked():
                display = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            elif self.radio_view_edge.isChecked():
                display = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)

            elif self.radio_view_lines.isChecked():
                display = original.copy()
                h, w = edge_map.shape

                # 수평선 (녹색)
                if h_clusters:
                    cluster = h_clusters[0]
                    a, b = cluster['coef1'], cluster['coef2']
                    x_start, x_end = 0, w - 1
                    y_start = int(a * x_start + b)
                    y_end = int(a * x_end + b)
                    y_start = max(0, min(h - 1, y_start))
                    y_end = max(0, min(h - 1, y_end))
                    cv2.line(display, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

                # 수직선 (파란색)
                if v_clusters:
                    cluster = v_clusters[0]
                    c, d = cluster['coef1'], cluster['coef2']
                    y_start, y_end = 0, h - 1
                    x_start = int(c * y_start + d)
                    x_end = int(c * y_end + d)
                    x_start = max(0, min(w - 1, x_start))
                    x_end = max(0, min(w - 1, x_end))
                    cv2.line(display, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            elif self.radio_view_overlay.isChecked():
                # 원본 + Edge overlay
                display = original.copy()
                edge_color = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
                edge_color[:, :, 0] = 0  # Blue channel = 0
                edge_color[:, :, 2] = 0  # Red channel = 0
                display = cv2.addWeighted(display, 0.7, edge_color, 0.3, 0)
                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            else:
                display = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        h, w, ch = display.shape
        bytes_per_line = ch * w
        q_img = QImage(display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.label_test_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_test_image.setPixmap(scaled)

    def test_prev_image(self):
        """이전 테스트 이미지"""
        if self.test_current_idx > 0:
            self.test_current_idx -= 1
            self.update_test_image_info()
            self.update_test_display()

            # 결과가 있으면 UI 업데이트
            img_path = self.test_images[self.test_current_idx]
            if img_path in self.test_results:
                result = self.test_results[img_path]
                self.label_h_lines_count.setText(f"{len(result['h_clusters'])} clusters ({len(result['h_lines'])} lines)")
                self.label_v_lines_count.setText(f"{len(result['v_clusters'])} clusters ({len(result['v_lines'])} lines)")
                self.label_inference_time.setText(f"{result['inference_time']:.1f} ms")

    def test_next_image(self):
        """다음 테스트 이미지"""
        if self.test_current_idx < len(self.test_images) - 1:
            self.test_current_idx += 1
            self.update_test_image_info()
            self.update_test_display()

            img_path = self.test_images[self.test_current_idx]
            if img_path in self.test_results:
                result = self.test_results[img_path]
                self.label_h_lines_count.setText(f"{len(result['h_clusters'])} clusters ({len(result['h_lines'])} lines)")
                self.label_v_lines_count.setText(f"{len(result['v_clusters'])} clusters ({len(result['v_lines'])} lines)")
                self.label_inference_time.setText(f"{result['inference_time']:.1f} ms")

    def save_current_result(self):
        """현재 결과 저장"""
        if not self.test_images:
            return

        img_path = self.test_images[self.test_current_idx]
        if img_path not in self.test_results:
            QMessageBox.warning(self, "Warning", "Run test first.")
            return

        output_folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", "result/"
        )
        if not output_folder:
            return

        result = self.test_results[img_path]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        cv2.imwrite(os.path.join(output_folder, f"{base_name}_edge.png"), result['edge_map'])

        lines_img = result['original'].copy()
        h, w = result['edge_map'].shape
        if result['h_clusters']:
            cluster = result['h_clusters'][0]
            a, b = cluster['coef1'], cluster['coef2']
            y_s = int(a * 0 + b)
            y_e = int(a * (w-1) + b)
            cv2.line(lines_img, (0, y_s), (w-1, y_e), (0, 255, 0), 2)
        if result['v_clusters']:
            cluster = result['v_clusters'][0]
            c, d = cluster['coef1'], cluster['coef2']
            x_s = int(c * 0 + d)
            x_e = int(c * (h-1) + d)
            cv2.line(lines_img, (x_s, 0), (x_e, h-1), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_lines.png"), lines_img)

        self.statusbar.showMessage(f"Saved: {base_name}")

    def show_final_lines(self):
        """최종 가로/세로 라인을 원본 이미지에 표시"""
        if not self.test_images:
            QMessageBox.warning(self, "Warning", "Load test images first.")
            return

        # 먼저 테스트 실행
        self.run_test()

        img_path = self.test_images[self.test_current_idx]
        if img_path not in self.test_results:
            return

        result = self.test_results[img_path]

        # 원본 이미지 복사
        lines_img = result['original'].copy()
        h, w = lines_img.shape[:2]

        # 가로 라인 (빨간색) - 1개만
        if result['h_clusters']:
            cluster = result['h_clusters'][0]
            a, b = cluster['coef1'], cluster['coef2']
            y_s = int(a * 0 + b)
            y_e = int(a * (w-1) + b)
            cv2.line(lines_img, (0, y_s), (w-1, y_e), (0, 0, 255), 2)

        # 세로 라인 (파란색) - 1개만
        if result['v_clusters']:
            cluster = result['v_clusters'][0]
            c, d = cluster['coef1'], cluster['coef2']
            x_s = int(c * 0 + d)
            x_e = int(c * (h-1) + d)
            cv2.line(lines_img, (x_s, 0), (x_e, h-1), (255, 0, 0), 2)

        # QLabel에 표시
        h_disp, w_disp = lines_img.shape[:2]
        bytes_per_line = 3 * w_disp
        q_img = QImage(lines_img.data, w_disp, h_disp, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)

        # label_test_image 크기에 맞춰 스케일
        scaled_pixmap = pixmap.scaled(
            self.label_test_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_test_image.setPixmap(scaled_pixmap)

        self.statusbar.showMessage(f"Final Lines: H={len(result['h_clusters'])}, V={len(result['v_clusters'])}")

    def batch_save_results(self):
        """모든 결과 일괄 저장"""
        if not self.test_results:
            QMessageBox.warning(self, "Warning", "No test results to save.")
            return

        output_folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", "result/"
        )
        if not output_folder:
            return

        count = 0
        for img_path, result in self.test_results.items():
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_edge.png"), result['edge_map'])

            lines_img = result['original'].copy()
            h, w = result['edge_map'].shape
            if result['h_clusters']:
                cluster = result['h_clusters'][0]
                a, b = cluster['coef1'], cluster['coef2']
                y_s = int(a * 0 + b)
                y_e = int(a * (w-1) + b)
                cv2.line(lines_img, (0, y_s), (w-1, y_e), (0, 255, 0), 2)
            if result['v_clusters']:
                cluster = result['v_clusters'][0]
                c, d = cluster['coef1'], cluster['coef2']
                x_s = int(c * 0 + d)
                x_e = int(c * (h-1) + d)
                cv2.line(lines_img, (x_s, 0), (x_e, h-1), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_lines.png"), lines_img)
            count += 1

        QMessageBox.information(
            self, "Batch Save Complete",
            f"Saved {count} results to:\n{output_folder}"
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
