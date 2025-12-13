"""
TEED Shared Memory Interface for C++ Communication

C++에서 전처리된 이미지를 공유 메모리로 받아 TEED 추론 후 결과 반환
- 입력: float32 tensor (3 x H x W) in CHW format
- 출력: uint8 edge map (H x W)

Windows API 호환 버전 (mmap + tagname)
"""

import os
import sys
import time
import struct
import numpy as np
import torch
import mmap

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teed_model import TED


class TEEDSharedMemory:
    """
    TEED 모델 공유 메모리 인터페이스 (Windows API 호환)

    C++ 프로그램과 공유 메모리를 통해 통신:
    - 입력 공유 메모리: C++ -> Python (전처리된 이미지)
    - 출력 공유 메모리: Python -> C++ (Edge map)
    """

    # 공유 메모리 이름
    SHM_INPUT_NAME = "teed_input"
    SHM_OUTPUT_NAME = "teed_output"

    # 헤더 크기 (height: 4bytes, width: 4bytes)
    HEADER_SIZE = 8

    def __init__(self, checkpoint_path="weights/prevision/best_model.pth", device=None):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로
            device: 'cuda' or 'cpu' (None이면 자동 선택)
        """
        # Device 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[TEED] Device: {self.device}")

        # 모델 로드
        self.model = TED().to(self.device)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[TEED] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[TEED] Warning: Checkpoint not found: {checkpoint_path}")

        self.model.eval()

        # 공유 메모리 (Windows mmap)
        self.mmap_input = None
        self.mmap_output = None
        self.input_size = 0
        self.output_size = 0
        self.input_shape = None  # (H, W)

    def connect(self, height=512, width=608):
        """
        공유 메모리 연결/생성 (Windows API 호환)

        Args:
            height: 이미지 높이 (8의 배수)
            width: 이미지 너비 (8의 배수)
        """
        self.input_shape = (height, width)

        # 입력 버퍼 크기: header + 3*H*W*4 (float32)
        self.input_size = self.HEADER_SIZE + 3 * height * width * 4

        # 출력 버퍼 크기: header + H*W (uint8)
        self.output_size = self.HEADER_SIZE + height * width

        # Windows mmap with tagname (C++ CreateFileMapping과 호환)
        # -1은 pagefile.sys를 사용 (INVALID_HANDLE_VALUE와 동일)
        self.mmap_input = mmap.mmap(-1, self.input_size, tagname=self.SHM_INPUT_NAME)
        print(f"[TEED] Created/opened input shm: {self.SHM_INPUT_NAME} ({self.input_size} bytes)")

        self.mmap_output = mmap.mmap(-1, self.output_size, tagname=self.SHM_OUTPUT_NAME)
        print(f"[TEED] Created/opened output shm: {self.SHM_OUTPUT_NAME} ({self.output_size} bytes)")

        # 헤더 초기화
        self.mmap_input.seek(0)
        self.mmap_input.write(struct.pack('ii', 0, 0))

        self.mmap_output.seek(0)
        self.mmap_output.write(struct.pack('ii', 0, 0))

        print(f"[TEED] Input shape: {height} x {width}")

    def disconnect(self):
        """공유 메모리 해제"""
        if self.mmap_input:
            self.mmap_input.close()
            self.mmap_input = None

        if self.mmap_output:
            self.mmap_output.close()
            self.mmap_output = None

        print("[TEED] Disconnected from shared memory")

    def read_input(self):
        """
        공유 메모리에서 입력 텐서 읽기

        Returns:
            tensor: (1, 3, H, W) float32 tensor
            height, width: 이미지 크기
        """
        if self.mmap_input is None:
            raise RuntimeError("Shared memory not connected")

        # 헤더 읽기
        self.mmap_input.seek(0)
        header_data = self.mmap_input.read(self.HEADER_SIZE)
        height, width = struct.unpack('ii', header_data)

        # 데이터 읽기 (float32, CHW format)
        data_size = 3 * height * width * 4
        data_bytes = self.mmap_input.read(data_size)
        data = np.frombuffer(data_bytes, dtype=np.float32).reshape(1, 3, height, width).copy()

        tensor = torch.from_numpy(data).to(self.device)

        return tensor, height, width

    def write_output(self, edge_map, height, width):
        """
        Edge map을 공유 메모리에 쓰기

        Args:
            edge_map: (H, W) uint8 numpy array
            height, width: 이미지 크기
        """
        if self.mmap_output is None:
            raise RuntimeError("Shared memory not connected")

        # 헤더 쓰기
        self.mmap_output.seek(0)
        self.mmap_output.write(struct.pack('ii', height, width))

        # 데이터 쓰기
        edge_bytes = edge_map.astype(np.uint8).tobytes()
        self.mmap_output.write(edge_bytes)

    def clear_input_header(self):
        """입력 헤더 초기화 (처리 완료 표시)"""
        self.mmap_input.seek(0)
        self.mmap_input.write(struct.pack('ii', 0, 0))

    def infer(self, tensor):
        """
        TEED 추론

        Args:
            tensor: (1, 3, H, W) float32 tensor

        Returns:
            edge_map: (H, W) uint8 numpy array (0-255)
        """
        with torch.no_grad():
            preds = self.model(tensor, single_test=True)

            # sigmoid + normalize
            pred = torch.sigmoid(preds[-1]).cpu().numpy()[0, 0]

            pred_min = pred.min()
            pred_max = pred.max()
            if pred_max > pred_min:
                pred = (pred - pred_min) / (pred_max - pred_min)

            edge_map = (pred * 255).astype(np.uint8)

        return edge_map

    def process_once(self):
        """
        한 번 처리: 입력 읽기 -> 추론 -> 출력 쓰기

        Returns:
            inference_time: 추론 시간 (ms)
        """
        # 입력 읽기
        tensor, height, width = self.read_input()

        # 추론
        t_start = time.perf_counter()
        edge_map = self.infer(tensor)
        t_end = time.perf_counter()
        inference_time = (t_end - t_start) * 1000

        # 출력 쓰기
        self.write_output(edge_map, height, width)

        return inference_time

    def run_service(self, poll_interval=0.001):
        """
        서비스 모드로 실행 (무한 루프)

        Args:
            poll_interval: 폴링 간격 (초)
        """
        print("[TEED] Service started. Press Ctrl+C to stop.")

        try:
            while True:
                # 입력 확인 (헤더만 읽어서 변경 감지)
                self.mmap_input.seek(0)
                header_data = self.mmap_input.read(self.HEADER_SIZE)
                height, width = struct.unpack('ii', header_data)

                # 유효한 입력이 있으면 처리
                if height > 0 and width > 0:
                    # 새로운 입력 감지
                    inference_time = self.process_once()
                    print(f"[TEED] Processed {width}x{height} in {inference_time:.1f}ms")

                    # 입력 헤더 초기화 (처리 완료 표시)
                    self.clear_input_header()

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n[TEED] Service stopped by user")


def main():
    """테스트/서비스 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='TEED Shared Memory Service')
    parser.add_argument('--checkpoint', type=str,
                        default='weights/prevision/best_model.pth',
                        help='Model checkpoint path')
    parser.add_argument('--height', type=int, default=512,
                        help='Input image height (default: 512)')
    parser.add_argument('--width', type=int, default=608,
                        help='Input image width (default: 608)')
    parser.add_argument('--test', action='store_true',
                        help='Run single test instead of service')
    args = parser.parse_args()

    # TEED 서비스 초기화
    teed = TEEDSharedMemory(checkpoint_path=args.checkpoint)
    teed.connect(height=args.height, width=args.width)

    try:
        if args.test:
            # 테스트 모드: 더미 입력으로 추론 테스트
            print("[TEED] Test mode: generating dummy input...")

            # 더미 입력 생성
            dummy_input = np.random.randn(3, args.height, args.width).astype(np.float32)

            # 공유 메모리에 쓰기
            teed.mmap_input.seek(0)
            teed.mmap_input.write(struct.pack('ii', args.height, args.width))
            teed.mmap_input.write(dummy_input.tobytes())

            # 처리
            inference_time = teed.process_once()
            print(f"[TEED] Test inference time: {inference_time:.1f}ms")

            # 출력 확인
            teed.mmap_output.seek(0)
            out_h, out_w = struct.unpack('ii', teed.mmap_output.read(teed.HEADER_SIZE))
            print(f"[TEED] Output shape: {out_w}x{out_h}")

        else:
            # 서비스 모드
            teed.run_service()

    finally:
        teed.disconnect()


if __name__ == '__main__':
    main()
