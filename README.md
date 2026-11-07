import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm


class U_Net:

    def __init__(self):

        # ==================== 1. 基础配置 ====================

        self.device = device

        self.height = 768
        self.width = 800

        # 35个网络：5行×7列
        self.num_rows = 5
        self.num_cols = 7

        self.x_starts = [
            0, 783, 1566, 2349, 3132, 3914, 4696
        ]

        self.y_starts = [
            0, 726, 1452, 2178, 2904
        ]

        # 原图尺寸
        self.H = 3672
        self.W = 5496

        # ==================== 2. 路径配置 ====================

        self.label_dir = r'D:\数据集\测试集\真实深度(灰度)_test\sharp'
        self.train_dir = r'D:\数据集\测试集\真实深度(灰度)_test\blur'

        self.exp_name = '6layer_3000_32_5轮'

        self.base_dir = r'D:\深度距离\深度距离训练结果\MIMOUNet_32_24_真实深度图'

        self.weight_dir = os.path.join(
            self.base_dir, 'weights_MIMO_32_24'
        )

        self.eval_dir = os.path.join(
            self.base_dir, 'evaluation_MIMO_32_24', 'test_results'
        )

        self.dir_pred = os.path.join(self.eval_dir, 'pred')
        self.dir_true = os.path.join(self.eval_dir, 'true')

        os.makedirs(self.dir_pred, exist_ok=True)
        os.makedirs(self.dir_true, exist_ok=True)

        # ==================== 3. 创建网络 ====================

        self.unet = MIMOUNet(
            channels=1,
            out_channels=1
        ).to(self.device)

    # =========================================================
    # 查找网络权重
    # =========================================================

    def get_model_path(self, row, col):

        network_name = f'Network_R{row+1:02d}_C{col+1:02d}'

        # 根据训练时的保存目录结构查找
        candidates = [
            os.path.join(
                self.weight_dir,
                self.exp_name,
                network_name,
                'best_model.pth'
            ),

            os.path.join(
                self.weight_dir,
                network_name,
                'best_model.pth'
            )
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f'找不到网络 {network_name} 的 best_model.pth\n'
            + '\n'.join(candidates)
        )

    # =========================================================
    # 加载网络
    # =========================================================

    def load_network(self, row, col):

        model_path = self.get_model_path(row, col)

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.unet.load_state_dict(checkpoint)

        self.unet.eval()

        print(
            f'加载网络 R{row+1:02d}_C{col+1:02d} 成功'
        )

    # =========================================================
    # 35个网络融合推理
    # =========================================================

    def predict_one_image(self, X):

        # X: [1, 1, H, W]

        _, _, H, W = X.shape

        if H != self.H or W != self.W:
            raise ValueError(
                f'图像尺寸错误：当前 {W}×{H}，'
                f'预期 {self.W}×{self.H}'
            )

        # 累加预测结果
        output_sum = np.zeros(
            (H, W),
            dtype=np.float32
        )

        # 记录每个像素被覆盖的次数
        weight_sum = np.zeros(
            (H, W),
            dtype=np.float32
        )

        # 逐个网络推理
        with torch.no_grad():

            for row in range(self.num_rows):

                for col in range(self.num_cols):

                    # 加载当前区域对应的网络
                    self.load_network(row, col)

                    x0 = self.x_starts[col]
                    y0 = self.y_starts[row]

                    x1 = x0 + self.width
                    y1 = y0 + self.height

                    # 裁剪当前网络对应区域
                    X_patch = X[:, :, y0:y1, x0:x1]

                    X_patch = X_patch.to(self.device)

                    # 网络推理
                    outputs = self.unet(X_patch)

                    # 完整分辨率输出
                    pred = outputs[2]

                    pred = pred.squeeze().float().cpu().numpy()

                    # 累加
                    output_sum[y0:y1, x0:x1] += pred

                    weight_sum[y0:y1, x0:x1] += 1.0

                    del outputs, pred, X_patch

                    torch.cuda.empty_cache()

        # 重叠区域取平均
        output_final = output_sum / np.maximum(
            weight_sum, 1e-8
        )

        return output_final

    # =========================================================
    # 测试
    # =========================================================

    def test(self, batch_size=1):

        test_dataset = MyDataset(
            self.train_dir,
            self.label_dir
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(f'测试集数量：{len(test_dataset)}')

        test_loss = 0.0
        test_count = 0

        for index, (Xb, Yb) in enumerate(
            tqdm(test_loader, desc='Testing')
        ):

            Xb = Xb.to(self.device)

            # 逐张图像推理
            for j in range(Xb.size(0)):

                X = Xb[j:j+1]

                Y = Yb[j, 0].numpy()

                # 35个网络推理和融合
                pred = self.predict_one_image(X)

                # 限制输出范围
                pred = np.clip(pred, 0.0, 1.0)

                # 计算 Charbonnier 测试损失
                diff = pred - Y

                loss = np.mean(
                    np.sqrt(diff ** 2 + 1e-6)
                )

                test_loss += loss
                test_count += 1

                # 保存预测图
                pred_img = Image.fromarray(
                    (pred * 255.0).astype(np.uint8)
                )

                pred_img.save(
                    os.path.join(
                        self.dir_pred,
                        f'{index * batch_size + j + 1}.png'
                    )
                )

                # 保存标签图
                true_img = Image.fromarray(
                    (np.clip(Y, 0, 1) * 255.0).astype(np.uint8)
                )

                true_img.save(
                    os.path.join(
                        self.dir_true,
                        f'{index * batch_size + j + 1}.png'
                    )
                )

        test_loss /= max(test_count, 1)

        print('\n========== 测试完成 ==========')
        print(f'Test Loss: {test_loss:.6f}')
        print(f'预测结果：{self.dir_pred}')
        print(f'标签结果：{self.dir_true}')


if __name__ == '__main__':

    unet = U_Net()

    unet.test(batch_size=1)
