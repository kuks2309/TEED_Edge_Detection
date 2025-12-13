"""
Test TEED model trained on PREVISION
"""
import os
import sys
import cv2
import numpy as np
import torch
from teed_model import TED
from teed_dataset import dataset_info

# 설정
TEST_DATA = 'PREVISION'
IS_LINUX = False
CHECKPOINT = 'weights/Prevision_MLCC/deploy/best_model.pth'
OUTPUT_DIR = 'result/prevision_new'


def extract_lines(edge_map, threshold=50, min_length=30, max_gap=10, angle_tolerance=30):
    """
    Edge map에서 수평/수직 직선 추출

    Args:
        edge_map: TEED 출력 (0-255 grayscale)
        threshold: 이진화 threshold
        min_length: 최소 선 길이
        max_gap: 선분 연결 최대 gap
        angle_tolerance: 수평/수직 허용 각도 범위 (기본 30도)

    Returns:
        horizontal_lines: [(x1, y1, x2, y2, length, angle, a, b), ...] 수평선 y=ax+b
        vertical_lines: [(x1, y1, x2, y2, length, angle, c, d), ...] 수직선 x=cy+d
    """
    # 이진화
    _, binary = cv2.threshold(edge_map, threshold, 255, cv2.THRESH_BINARY)

    # Hough Line Transform
    hough_lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=min_length,
        maxLineGap=max_gap
    )

    horizontal_lines = []
    vertical_lines = []

    if hough_lines is not None:
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi  # -180 ~ 180

            # 수평선: -30 ~ 30도, y = ax + b
            if -angle_tolerance <= angle <= angle_tolerance:
                if x2 != x1:
                    a = (y2 - y1) / (x2 - x1)
                    b = y1 - a * x1
                else:
                    a = 0
                    b = (y1 + y2) / 2
                horizontal_lines.append((x1, y1, x2, y2, length, angle, a, b))

            # 수직선: 60~120도 또는 -60~-120도, x = cy + d
            elif (90 - angle_tolerance) <= abs(angle) <= (90 + angle_tolerance):
                if y2 != y1:
                    c = (x2 - x1) / (y2 - y1)
                    d = x1 - c * y1
                else:
                    c = 0
                    d = (x1 + x2) / 2
                vertical_lines.append((x1, y1, x2, y2, length, angle, c, d))

    # 길이순 정렬
    horizontal_lines.sort(key=lambda x: x[4], reverse=True)
    vertical_lines.sort(key=lambda x: x[4], reverse=True)

    return horizontal_lines, vertical_lines


def cluster_lines(lines, is_horizontal=True, threshold=10):
    """
    직선들을 방정식 계수로 클러스터링

    Args:
        lines: 직선 리스트
        is_horizontal: True면 수평선(y=ax+b), False면 수직선(x=cy+d)
        threshold: 클러스터링 거리 threshold (픽셀)

    Returns:
        clusters: [{'lines': [...], 'a': avg_a, 'b': avg_b, 'total_length': ...}, ...]
    """
    if not lines:
        return []

    # 직선을 (기울기, 절편) 기준으로 클러스터링
    clusters = []

    for line in lines:
        x1, y1, x2, y2, length, angle, coef1, coef2 = line

        # 기존 클러스터와 비교
        merged = False
        for cluster in clusters:
            # 절편 차이로 클러스터링 (기울기가 비슷하므로 절편으로 구분)
            if abs(cluster['coef2'] - coef2) < threshold:
                cluster['lines'].append(line)
                cluster['total_length'] += length
                # 가중 평균으로 계수 업데이트
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

    # 총 길이순 정렬
    clusters.sort(key=lambda x: x['total_length'], reverse=True)

    return clusters


def main():
    print("=" * 50)
    print("TEED Test on PREVISION")
    print("=" * 50)

    # 명령줄 인자로 이미지 폴더 지정 가능
    if len(sys.argv) > 1:
        imgs_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'result/custom'
    else:
        test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
        data_dir = test_inf['data_dir']
        imgs_dir = os.path.join(data_dir, 'imgs', 'train')
        output_dir = OUTPUT_DIR

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Test images: {imgs_dir}")

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Model
    model = TED().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # Mean values for normalization
    mean = [104.007, 116.669, 122.679]  # PREVISION mean

    # Get test images (PNG, JPG, BMP)
    image_files = [f for f in os.listdir(imgs_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Found {len(image_files)} images")

    with torch.no_grad():
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(imgs_dir, img_name)

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue

            h_orig, w_orig = img.shape[:2]

            # 이미지 크기에 따라 처리 방식 결정
            if h_orig > 1000:  # 원본 크기 (2448x2048 등)
                # 원본을 32의 배수로 crop 후 1/4 축소
                h_crop = (h_orig // 32) * 32
                w_crop = (w_orig // 32) * 32
                y_start = (h_orig - h_crop) // 2
                x_start = (w_orig - w_crop) // 2
                img = img[y_start:y_start+h_crop, x_start:x_start+w_crop]
                h_new = h_crop // 4
                w_new = w_crop // 4
                img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
            else:  # 이미 축소된 이미지 (512x612 등)
                # 8의 배수로 crop만 수행
                h_new = (h_orig // 8) * 8
                w_new = (w_orig // 8) * 8
                y_start = (h_orig - h_new) // 2
                x_start = (w_orig - w_new) // 2
                img = img[y_start:y_start+h_new, x_start:x_start+w_new]

            # Preprocess (BGR -> normalized)
            img_tensor = img.astype(np.float32)
            img_tensor -= mean
            img_tensor = img_tensor.transpose((2, 0, 1))  # HWC -> CHW
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)

            # Inference
            preds = model(img_tensor, single_test=True)

            # Get final output
            pred = torch.sigmoid(preds[-1]).cpu().numpy()[0, 0]

            # 정규화: 배경을 0으로, edge를 255로 맞춤
            pred_min = pred.min()
            pred_max = pred.max()
            if pred_max > pred_min:
                pred = (pred - pred_min) / (pred_max - pred_min)
            pred = (pred * 255).astype(np.uint8)

            # 직선 추출 (수평/수직 분리)
            h_lines, v_lines = extract_lines(pred, threshold=50, min_length=30, max_gap=10, angle_tolerance=30)

            # 클러스터링 (절편 기준 10픽셀 이내 동일 직선으로 간주)
            h_clusters = cluster_lines(h_lines, is_horizontal=True, threshold=10)
            v_clusters = cluster_lines(v_lines, is_horizontal=False, threshold=10)

            # 결과 이미지 생성 (가장 큰 클러스터 1개씩만)
            result_img = img.copy()

            # 수평선: 가장 큰 클러스터 1개 (녹색)
            if h_clusters:
                cluster = h_clusters[0]  # 가장 큰 클러스터
                a, b = cluster['coef1'], cluster['coef2']
                x_start, x_end = 0, w_new - 1
                y_start = int(a * x_start + b)
                y_end = int(a * x_end + b)
                y_start = max(0, min(h_new - 1, y_start))
                y_end = max(0, min(h_new - 1, y_end))
                cv2.line(result_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

            # 수직선: 가장 큰 클러스터 1개 (파란색)
            if v_clusters:
                cluster = v_clusters[0]  # 가장 큰 클러스터
                c, d = cluster['coef1'], cluster['coef2']
                y_start, y_end = 0, h_new - 1
                x_start = int(c * y_start + d)
                x_end = int(c * y_end + d)
                x_start = max(0, min(w_new - 1, x_start))
                x_end = max(0, min(w_new - 1, x_end))
                cv2.line(result_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

            # Save results
            base_name = os.path.splitext(img_name)[0]

            # 1. Edge map 저장
            edge_path = os.path.join(output_dir, f"{base_name}_edge.png")
            cv2.imwrite(edge_path, pred)

            # 2. 클러스터링된 직선 결과 저장
            lines_path = os.path.join(output_dir, f"{base_name}_lines.png")
            cv2.imwrite(lines_path, result_img)

            print(f"[{i+1}/{len(image_files)}] {base_name}: H={len(h_clusters)} clusters ({len(h_lines)} lines), V={len(v_clusters)} clusters ({len(v_lines)} lines)")

    print("=" * 50)
    print(f"Test complete! Results saved to: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
