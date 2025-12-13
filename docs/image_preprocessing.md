# TEED 이미지 전처리 가이드

## 개요

TEED 모델은 내부적으로 2배씩 downsampling을 수행하므로 입력 이미지 크기가 **8의 배수**여야 합니다.
이 문서는 효율적인 이미지 전처리 방법을 설명합니다.

## 이미지 크기 요구사항

| 항목 | 값 |
|-----|---|
| 입력 크기 | 8의 배수 (H, W 모두) |
| 권장 크기 | 512 x 608 (2448x2048 원본의 1/4) |
| 채널 | BGR (OpenCV 기본) |

## 전처리 방식 비교

### 기존 방식 (비권장)

```python
# 1단계: 1/4 축소
h_new = int(h_orig * 0.25)  # 2048 -> 512
w_new = int(w_orig * 0.25)  # 2448 -> 612

# 2단계: 8의 배수로 조정 (추가 resize 필요)
h_new = (h_new // 8) * 8    # 512 -> 512
w_new = (w_new // 8) * 8    # 612 -> 608
img = cv2.resize(img, (w_new, h_new))
```

**문제점:**
- resize 2번 수행
- 612 -> 608 추가 조정 필요

### 새 방식 (권장)

```python
# 1단계: 원본을 32의 배수로 crop (1/4 후 8의 배수 보장)
h_crop = (h_orig // 32) * 32  # 2048 -> 2048
w_crop = (w_orig // 32) * 32  # 2448 -> 2432

# 중앙 기준 crop
y_start = (h_orig - h_crop) // 2
x_start = (w_orig - w_crop) // 2
img = img[y_start:y_start+h_crop, x_start:x_start+w_crop]

# 2단계: 1/4 축소 (자동으로 8의 배수)
h_new = h_crop // 4  # 2048 -> 512
w_new = w_crop // 4  # 2432 -> 608
img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
```

**장점:**
- resize 1번만 수행
- crop은 메모리 인덱싱만 (거의 0ms)
- 정수 나눗셈으로 깔끔한 계산

## 수학적 원리

```
원본 크기를 32의 배수로 crop하면:
- 32 = 8 × 4
- crop된 크기 / 4 = 자동으로 8의 배수

예시: 2448 × 2048
- 2448 // 32 × 32 = 2432 (16픽셀 crop)
- 2048 // 32 × 32 = 2048 (crop 없음)
- 2432 / 4 = 608 (8의 배수)
- 2048 / 4 = 512 (8의 배수)
```

## 성능 비교

| 방식 | 처리 시간 | 속도 향상 |
|-----|----------|----------|
| 기존 (resize 2번) | 1.31 ms | - |
| 새것 (crop + resize 1번) | 0.95 ms | **28% 향상** |

## 전처리 전체 파이프라인

```python
import cv2
import numpy as np

def preprocess_image(img, mean=[104.007, 116.669, 122.679]):
    """
    TEED 모델용 이미지 전처리

    Args:
        img: BGR 이미지 (numpy array)
        mean: BGR 채널별 평균값

    Returns:
        tensor: 전처리된 텐서 (1, 3, H, W)
        h_new, w_new: 출력 크기
    """
    h_orig, w_orig = img.shape[:2]

    # 1. 32의 배수로 중앙 crop
    h_crop = (h_orig // 32) * 32
    w_crop = (w_orig // 32) * 32
    y_start = (h_orig - h_crop) // 2
    x_start = (w_orig - w_crop) // 2
    img = img[y_start:y_start+h_crop, x_start:x_start+w_crop]

    # 2. 1/4 축소
    h_new = h_crop // 4
    w_new = w_crop // 4
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)

    # 3. 정규화
    img_tensor = img.astype(np.float32)
    img_tensor -= mean

    # 4. HWC -> CHW -> NCHW
    img_tensor = img_tensor.transpose((2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)

    return img_tensor, h_new, w_new
```

## 출력 후처리

```python
def postprocess_edge(pred, h_new, w_new):
    """
    TEED 출력 후처리

    Args:
        pred: 모델 출력 텐서
        h_new, w_new: 출력 크기

    Returns:
        edge_map: 0-255 grayscale edge map
    """
    import torch

    # sigmoid 적용
    pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

    # min-max 정규화
    pred_min = pred.min()
    pred_max = pred.max()
    if pred_max > pred_min:
        pred = (pred - pred_min) / (pred_max - pred_min)

    # 0-255 스케일
    edge_map = (pred * 255).astype(np.uint8)

    return edge_map
```

## 원본 좌표 변환

축소된 이미지에서 검출된 좌표를 원본 좌표로 변환:

```python
def to_original_coords(x, y, scale=4, x_offset=0, y_offset=0):
    """
    축소 이미지 좌표 -> 원본 좌표 변환

    Args:
        x, y: 축소 이미지에서의 좌표
        scale: 축소 비율 (기본 4)
        x_offset, y_offset: crop 오프셋

    Returns:
        x_orig, y_orig: 원본 이미지 좌표
    """
    x_orig = x * scale + x_offset
    y_orig = y * scale + y_offset
    return x_orig, y_orig
```

## C++ 구현 (공유 메모리 연동용)

실제 운영 환경에서는 **C++에서 전처리를 수행**하고, Python은 TEED 추론만 담당합니다.
이는 처리 속도 최적화와 공유 메모리 통신 효율성을 위함입니다.

### 시스템 구조

```
[C++ 프로그램]                    [Python TEED]
     │                                │
     ├── 이미지 획득                   │
     ├── 32배수 crop                  │
     ├── 1/4 resize                   │
     ├── BGR 정규화 (mean 빼기)        │
     ├── HWC -> CHW 변환              │
     │                                │
     └── 공유 메모리 Write ──────────> 공유 메모리 Read
                                      │
                                      ├── TEED 추론 (~48ms)
                                      │
     공유 메모리 Read <────────────── 공유 메모리 Write
     │
     ├── Edge map 수신
     ├── Hough 직선 검출
     ├── 클러스터링
     └── 결과 처리
```

### C++ 전처리 코드 예시

```cpp
#include <opencv2/opencv.hpp>

struct PreprocessResult {
    std::vector<float> tensor;  // CHW format
    int height;
    int width;
};

PreprocessResult preprocess_for_teed(const cv::Mat& img) {
    // BGR mean values
    const float mean_b = 104.007f;
    const float mean_g = 116.669f;
    const float mean_r = 122.679f;

    int h_orig = img.rows;
    int w_orig = img.cols;

    // 1. 32의 배수로 crop
    int h_crop = (h_orig / 32) * 32;
    int w_crop = (w_orig / 32) * 32;
    int y_start = (h_orig - h_crop) / 2;
    int x_start = (w_orig - w_crop) / 2;

    cv::Mat cropped = img(cv::Rect(x_start, y_start, w_crop, h_crop));

    // 2. 1/4 축소
    int h_new = h_crop / 4;
    int w_new = w_crop / 4;
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(w_new, h_new), 0, 0, cv::INTER_AREA);

    // 3. float 변환 및 정규화, CHW 순서로 저장
    PreprocessResult result;
    result.height = h_new;
    result.width = w_new;
    result.tensor.resize(3 * h_new * w_new);

    // CHW 순서: B채널 전체 -> G채널 전체 -> R채널 전체
    for (int y = 0; y < h_new; y++) {
        for (int x = 0; x < w_new; x++) {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
            int idx = y * w_new + x;

            // Channel 0: Blue
            result.tensor[0 * h_new * w_new + idx] = pixel[0] - mean_b;
            // Channel 1: Green
            result.tensor[1 * h_new * w_new + idx] = pixel[1] - mean_g;
            // Channel 2: Red
            result.tensor[2 * h_new * w_new + idx] = pixel[2] - mean_r;
        }
    }

    return result;
}
```

### 공유 메모리 데이터 포맷

**입력 (C++ -> Python):**
```
[Header: 8 bytes]
  - height: int32 (4 bytes)
  - width:  int32 (4 bytes)
[Data: 3 * H * W * 4 bytes]
  - float32 tensor in CHW format
  - 총 크기: 3 * 512 * 608 * 4 = 3,735,552 bytes (~3.6MB)
```

**출력 (Python -> C++):**
```
[Header: 8 bytes]
  - height: int32 (4 bytes)
  - width:  int32 (4 bytes)
[Data: H * W bytes]
  - uint8 edge map (0-255)
  - 총 크기: 512 * 608 = 311,296 bytes (~304KB)
```

### C++ 후처리 (직선 검출)

```cpp
#include <opencv2/opencv.hpp>

struct LineInfo {
    int x1, y1, x2, y2;
    float length;
    float angle;
    float coef1, coef2;  // y=ax+b (수평) 또는 x=cy+d (수직)
};

void extract_lines(const cv::Mat& edge_map,
                   std::vector<LineInfo>& h_lines,
                   std::vector<LineInfo>& v_lines,
                   int threshold = 50,
                   int min_length = 30,
                   int max_gap = 10,
                   float angle_tolerance = 30.0f) {

    // 이진화
    cv::Mat binary;
    cv::threshold(edge_map, binary, threshold, 255, cv::THRESH_BINARY);

    // Hough Line Transform
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(binary, lines, 1, CV_PI/180, 50, min_length, max_gap);

    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1];
        int x2 = line[2], y2 = line[3];

        float length = std::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
        float angle = std::atan2(y2-y1, x2-x1) * 180.0f / CV_PI;

        LineInfo info = {x1, y1, x2, y2, length, angle, 0, 0};

        // 수평선: -30 ~ 30도
        if (std::abs(angle) <= angle_tolerance) {
            if (x2 != x1) {
                info.coef1 = (float)(y2 - y1) / (x2 - x1);  // a
                info.coef2 = y1 - info.coef1 * x1;          // b
            } else {
                info.coef1 = 0;
                info.coef2 = (y1 + y2) / 2.0f;
            }
            h_lines.push_back(info);
        }
        // 수직선: 60~120도
        else if (std::abs(angle) >= (90 - angle_tolerance) &&
                 std::abs(angle) <= (90 + angle_tolerance)) {
            if (y2 != y1) {
                info.coef1 = (float)(x2 - x1) / (y2 - y1);  // c
                info.coef2 = x1 - info.coef1 * y1;          // d
            } else {
                info.coef1 = 0;
                info.coef2 = (x1 + x2) / 2.0f;
            }
            v_lines.push_back(info);
        }
    }

    // 길이순 정렬
    std::sort(h_lines.begin(), h_lines.end(),
              [](const LineInfo& a, const LineInfo& b) {
                  return a.length > b.length;
              });
    std::sort(v_lines.begin(), v_lines.end(),
              [](const LineInfo& a, const LineInfo& b) {
                  return a.length > b.length;
              });
}
```

## 주의사항

1. **학습/추론 일관성**: 학습과 추론 시 동일한 전처리 적용 필요
2. **mean 값**: 데이터셋에 맞는 mean 값 사용 (MYDATA: [104.007, 116.669, 122.679])
3. **crop 손실**: 가장자리 최대 31픽셀 손실 가능 (중요 영역이 가장자리에 있으면 주의)
4. **C++/Python 데이터 정합성**: CHW 포맷, float32 타입, BGR 순서 일치 확인 필수
