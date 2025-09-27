#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线训练脚本：使用 Neural-Fly 提供的 Phi_Net 结构训练 DAIML 表征。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
torch.set_default_dtype(torch.double)
import torch.nn.functional as F
from torch import optim

import utils  # type: ignore
from mlmodel import Phi_Net, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Neural-Fly Phi_Net 表征网络')
    parser.add_argument('--data-folder', type=str, default='data/experiment',
                        help='包含 csv 数据的目录，支持多个文件')
    parser.add_argument('--hover-ratio', type=float, default=1.0,
                        help='hover_pwm_ratio，跨平台数据缩放')
    parser.add_argument('--support-size', type=int, default=16,
                        help='每个任务 support 集大小 (K)')
    parser.add_argument('--query-size', type=int, default=64,
                        help='每个任务 query 集大小 (B)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='φ(x) 的维度 (不含 bias)')
    parser.add_argument('--ridge-lambda', type=float, default=1e-3,
                        help='support 闭式回归的正则系数')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--model-name', type=str, default='phi_sim',
                        help='保存模型名称 (./models/<model-name>.pth)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val-tasks', type=int, default=0,
                        help='每个 epoch 额外用于验证的任务数量（0 表示仅训练损失）')
    parser.add_argument('--log-json', type=str, default='',
                        help='可选：把训练日志写入 json 文件')
    return parser.parse_args()


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_datasets(data_folder: str, hover_ratio: float):
    raw = utils.load_data(data_folder)
    datasets = utils.format_data(raw, features=['v', 'q', 'pwm'], output='fa',
                                hover_pwm_ratio=hover_ratio)
    if len(datasets) == 0:
        raise RuntimeError(f'未在 {data_folder} 找到可用数据')
    dim_x = datasets[0].X.shape[1]
    dim_y = datasets[0].Y.shape[1]
    return datasets, dim_x, dim_y


def split_support_query(dataset, support_size: int, query_size: int):
    n = dataset.X.shape[0]
    total = support_size + query_size
    if n < total:
        raise ValueError(f'Task samples {n} < support+query {total}')
    idx = np.random.choice(n, total, replace=False)
    support_idx = idx[:support_size]
    query_idx = idx[support_size:]
    X = torch.from_numpy(dataset.X)
    Y = torch.from_numpy(dataset.Y)
    support_input = X[support_idx]
    support_label = Y[support_idx]
    query_input = X[query_idx]
    query_label = Y[query_idx]
    return support_input, support_label, query_input, query_label


def solve_ridge(phi_support: torch.Tensor, y_support: torch.Tensor, lam: float):
    dim_a = phi_support.shape[1]
    eye = torch.eye(dim_a, dtype=phi_support.dtype, device=phi_support.device)
    lhs = phi_support.T @ phi_support + lam * eye
    rhs = phi_support.T @ y_support
    a = torch.linalg.solve(lhs, rhs)
    return a


def meta_training_loop(*, datasets: List, phi_net: Phi_Net, options: dict, args: argparse.Namespace):
    device = torch.device(args.device)
    phi_net = phi_net.to(device)
    optimizer = optim.Adam(phi_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(datasets)
        epoch_loss = 0.0
        task_count = 0
        for task in datasets:
            if task.X.shape[0] < args.support_size + args.query_size:
                continue
            support_input, support_label, query_input, query_label = split_support_query(task, args.support_size, args.query_size)
            support_input = support_input.to(device)
            support_label = support_label.to(device)
            query_input = query_input.to(device)
            query_label = query_label.to(device)

            phi_support = phi_net(support_input)
            a = solve_ridge(phi_support, support_label, args.ridge_lambda)
            predictions = phi_net(query_input) @ a
            loss = F.mse_loss(predictions, query_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            task_count += 1

        epoch_loss /= max(task_count, 1)

        log_entry = {'epoch': epoch, 'train_loss': epoch_loss}

        if args.val_tasks > 0:
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for task in random.sample(datasets, min(args.val_tasks, len(datasets))):
                    if task.X.shape[0] < args.support_size + args.query_size:
                        continue
                    support_input, support_label, query_input, query_label = split_support_query(task, args.support_size, args.query_size)
                    support_input = support_input.to(device)
                    support_label = support_label.to(device)
                    query_input = query_input.to(device)
                    query_label = query_label.to(device)

                    phi_support = phi_net(support_input)
                    a = solve_ridge(phi_support, support_label, args.ridge_lambda)
                    predictions = phi_net(query_input) @ a
                    val_loss += float(F.mse_loss(predictions, query_label).item())
                    val_samples += 1
            if val_samples > 0:
                val_loss /= val_samples
                log_entry['val_loss'] = val_loss
        history.append(log_entry)
        rospy_like_log(f"[train_phi] epoch {epoch:03d} | train_loss {epoch_loss:.6f}" +
                       (f" | val_loss {log_entry['val_loss']:.6f}" if 'val_loss' in log_entry else ''))

    if args.log_json:
        Path(args.log_json).write_text(json.dumps(history, indent=2), encoding='utf-8')

    return phi_net.cpu(), history


def rospy_like_log(msg: str):
    print(msg, flush=True)


def main():
    args = parse_args()
    set_random_seed(args.seed)

    datasets, dim_x, dim_y = load_datasets(args.data_folder, args.hover_ratio)
    dim_a = args.latent_dim + 1

    options = {
        'dim_x': dim_x,
        'dim_y': dim_y,
        'dim_a': dim_a,
        'support_size': args.support_size,
        'query_size': args.query_size,
        'ridge_lambda': args.ridge_lambda,
        'loss_type': 'none',
    }

    phi_net = Phi_Net(options)
    phi_net, history = meta_training_loop(datasets=datasets,
                                          phi_net=phi_net,
                                          options=options,
                                          args=args)

    save_model(phi_net=phi_net, h_net=None, modelname=args.model_name, options=options)
    rospy_like_log(f"[train_phi] 模型已保存至 ./models/{args.model_name}.pth")


if __name__ == '__main__':
    main()
