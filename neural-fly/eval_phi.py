#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型评估脚本：加载训练好的 φ_net，计算多个数据集上的残差力误差统计。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import utils  # type: ignore
from mlmodel import load_model, error_statistics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained Neural-Fly Phi model')
    parser.add_argument('--model-name', type=str, required=True,
                        help='模型名称（对应 ./models/<model-name>.pth）')
    parser.add_argument('--data-folder', type=str, default='data/experiment',
                        help='评估数据所在目录')
    parser.add_argument('--hover-ratio', type=float, default=1.0,
                        help='hover_pwm_ratio 缩放因子')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model_name)
    phi_net = model.phi
    phi_net.eval()
    device = torch.device(args.device)
    phi_net.to(device)

    datasets = utils.format_data(utils.load_data(args.data_folder),
                                 features=['v', 'q', 'pwm'],
                                 output='fa',
                                 hover_pwm_ratio=args.hover_ratio)
    if len(datasets) == 0:
        raise RuntimeError('未找到任何评估数据')

    all_errors = []
    for data in datasets:
        X = torch.from_numpy(data.X).to(device)
        Y = torch.from_numpy(data.Y).to(device)
        err = []
        with torch.no_grad():
            phi = phi_net(X)
            A = torch.linalg.solve(phi.T @ phi + 1e-3 * torch.eye(phi.shape[1], device=device), phi.T @ Y)
            pred = phi @ A
            mse = torch.mean((pred - Y) ** 2, dim=0)
            err = mse.cpu().numpy()
        all_errors.append({'meta': data.meta, 'mse': err})
        print(f"{data.meta['method']}/{data.meta['condition']} -> MSE {err}")

    # 计算总体误差
    merged_X = np.vstack([d.X for d in datasets])
    merged_Y = np.vstack([d.Y for d in datasets])
    e1, e2, e3 = error_statistics(merged_X, merged_Y, phi_net, None, model.options)
    print('\n汇总误差:')
    print(f'  baseline zero   : {e1:.6f}')
    print(f'  average thrust  : {e2:.6f}')
    print(f'  phi adaptation  : {e3:.6f}')

    summary_path = Path(f'./models/{args.model_name}_eval.txt')
    with summary_path.open('w', encoding='utf-8') as f:
        for item in all_errors:
            f.write(f"{item['meta']} => {item['mse']}\n")
        f.write(f"\nbaseline_zero={e1}\naverage_output={e2}\nphi_adapt={e3}\n")
    print(f'评估结果已写入 {summary_path}')


if __name__ == '__main__':
    main()
