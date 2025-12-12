"""
TEED Training Script for MYDATA (Custom Dataset)
"""
from __future__ import print_function

import os
import time
import cv2
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from loss2 import cats_loss, bdcn_loss2
from ted import TED
from utils.img_processing import visualize_result, count_parameters

# 설정
TRAIN_DATA = 'MYDATA'  # 커스텀 데이터셋
TEST_DATA = 'MYDATA'
IS_LINUX = False

# 하이퍼파라미터
EPOCHS = 30
BATCH_SIZE = 2  # 작은 데이터셋이므로 작게
LR = 5e-4
WD = 2e-4
IMG_SIZE = 300  # crop size


def train_one_epoch(epoch, dataloader, model, criterions, optimizer, device):
    criterion1, criterion2 = criterions
    model.train()

    l_weight0 = [1.1, 0.7, 1.1, 1.3]
    l_weight = [[0.05, 2.], [0.05, 2.], [0.01, 1.], [0.01, 3.]]
    loss_avg = []

    for batch_id, sample_batched in enumerate(dataloader):
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

        if batch_id % 5 == 0:
            print(f'{time.ctime()} Epoch: {epoch} Batch {batch_id}/{len(dataloader)} Loss: {tLoss.item():.4f}')

    return np.array(loss_avg).mean()


def validate(dataloader, model, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']

            preds = model(images, single_test=True)

            # 마지막 출력 저장
            pred = torch.sigmoid(preds[-1]).cpu().numpy()[0, 0]

            # 정규화: 배경을 0으로, edge를 255로 맞춤
            pred_min = pred.min()
            pred_max = pred.max()
            if pred_max > pred_min:
                pred = (pred - pred_min) / (pred_max - pred_min)
            pred = (pred * 255).astype(np.uint8)

            # 원본 크기로 복원
            h, w = image_shape[0].item(), image_shape[1].item()
            pred = cv2.resize(pred, (w, h))

            save_path = os.path.join(output_dir, file_names[0])
            cv2.imwrite(save_path, pred)
            print(f'Saved: {save_path}')


def main():
    print("=" * 50)
    print("TEED Training on MYDATA")
    print("=" * 50)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset info
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)

    print(f"Train data: {train_inf['data_dir']}")
    print(f"Train list: {train_inf['train_list']}")

    # Output directory
    output_dir = os.path.join('checkpoints', TRAIN_DATA)
    os.makedirs(output_dir, exist_ok=True)

    # Model
    model = TED().to(device)

    # 기존 BIPED 체크포인트로 초기화 (Transfer Learning)
    pretrained_path = 'checkpoints/BIPED/5/5_model.pth'
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))

    num_param = count_parameters(model)
    print(f"Model parameters: {num_param}")

    # Dataset
    class Args:
        mean_train = train_inf['mean'][:3]
        mean_test = test_inf['mean'][:3]
        train_data = TRAIN_DATA
        train_list = train_inf['train_list']
        test_list = test_inf['test_list']
        test_img_width = test_inf['img_width']
        test_img_height = test_inf['img_height']
        up_scale = False

    args = Args()

    dataset_train = BipedDataset(
        train_inf['data_dir'],
        img_width=IMG_SIZE,
        img_height=IMG_SIZE,
        train_mode='train',
        arg=args
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Windows에서는 0으로
    )

    dataset_val = TestDataset(
        test_inf['data_dir'],
        test_data=TEST_DATA,
        img_width=test_inf['img_width'],
        img_height=test_inf['img_height'],
        test_list=test_inf['test_list'],
        arg=args
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"Train samples: {len(dataset_train)}")
    print(f"Val samples: {len(dataset_val)}")

    # Loss & Optimizer
    criterion1 = cats_loss
    criterion2 = bdcn_loss2
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # Training loop
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*50}")

        # Train
        avg_loss = train_one_epoch(
            epoch, dataloader_train, model,
            [criterion1, criterion2], optimizer, device
        )
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # Validate & Save
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            val_dir = os.path.join(output_dir, f'epoch_{epoch + 1}')
            validate(dataloader_val, model, device, val_dir)

            # Save checkpoint
            ckpt_path = os.path.join(output_dir, f'{epoch + 1}_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
