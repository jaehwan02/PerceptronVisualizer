import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class SimpleMLP(nn.Module):
    """
    입력: 8x8=64
    은닉 레이어: 16 -> 16 -> 16
    출력: 2 (1 or 2)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 2)  # 최종 출력 (2차원: label 0 or 1)

        # forward hook으로 각 레이어 출력(활성화값) 저장
        self.activations = {}

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook

        # hook 등록
        self.fc1.register_forward_hook(get_activation('hidden1'))
        self.fc2.register_forward_hook(get_activation('hidden2'))
        self.fc3.register_forward_hook(get_activation('hidden3'))
        self.fc4.register_forward_hook(get_activation('output'))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # (N,2)
        return x

    def visualize_activations(self):
        """
        은닉 레이어(3개) + 출력 레이어(2차원) 시각화
        """
        # 단일 배치(N=1) 가정
        h1 = self.activations['hidden1'][0]  # shape (16,)
        h2 = self.activations['hidden2'][0]  # shape (16,)
        h3 = self.activations['hidden3'][0]  # shape (16,)
        out= self.activations['output'][0]   # shape (2,)

        w1 = self.fc1.weight.detach().cpu().numpy()  # (16,64)
        w2 = self.fc2.weight.detach().cpu().numpy()  # (16,16)
        w3 = self.fc3.weight.detach().cpu().numpy()  # (16,16)
        w4 = self.fc4.weight.detach().cpu().numpy()  # (2,16)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim([0, 4])
        ax.set_ylim([5, 0])
        ax.axis('off')

        # 노드 좌표
        node_coords = {
            'input':   [(0.2, 2.5)],  # 입력 (64D) 축소 노드
            'hidden1': [(1, 0.3 + 3.2*(i/11.0)) for i in range(16)],
            'hidden2': [(2, 0.3 + 3.2*(i/11.0)) for i in range(16)],
            'hidden3': [(3, 0.3 + 3.2*(i/11.0)) for i in range(16)],
            'output':  [(3.8, 1.25+0.2), (3.8, 3.75-0.2)]  # 2개 (class 1,2)
        }

        # --- 엣지(선) ---
        # input->h1
        for i1, (hx, hy) in enumerate(node_coords['hidden1']):
            mean_abs = np.mean(np.abs(w1[i1, :]))
            lw = mean_abs * 5  # 선 두께 조정
            alpha = min(0.3 + mean_abs / 2, 1.0)  # 투명도 조정
            ax.plot([node_coords['input'][0][0], hx],
                    [node_coords['input'][0][1], hy],
                    color='black', linewidth=lw, alpha=alpha)

        # hidden1->hidden2
        for i2, (x2, y2) in enumerate(node_coords['hidden2']):
            for i1, (x1, y1) in enumerate(node_coords['hidden1']):
                w_val = w2[i2, i1]
                lw = abs(w_val) * 5  # 선 두께
                alpha = min(0.3 + abs(w_val) / 2, 1.0)  # 투명도
                ax.plot([x1, x2], [y1, y2], color='black', linewidth=lw, alpha=alpha)

        # hidden2->hidden3
        for i3, (x3, y3) in enumerate(node_coords['hidden3']):
            for i2_, (x2, y2) in enumerate(node_coords['hidden2']):
                w_val = w3[i3, i2_]
                lw = abs(w_val) * 5
                alpha = min(0.3 + abs(w_val) / 2, 1.0)
                ax.plot([x2, x3], [y2, y3], color='black', linewidth=lw, alpha=alpha)

        # hidden3->output
        for out_idx, (ox, oy) in enumerate(node_coords['output']):
            for i3_, (xx, yy) in enumerate(node_coords['hidden3']):
                w_val = w4[out_idx, i3_]
                lw = abs(w_val) * 5
                alpha = min(0.3 + abs(w_val) / 2, 1.0)
                ax.plot([xx, ox], [yy, oy], color='black', linewidth=lw, alpha=alpha)


        # helper function to plot hidden nodes
        def plot_hidden(acts, coords, ms=8, fs=12):
            a_min, a_max = acts.min(), acts.max()
            a_range = (a_max - a_min) if a_max!=a_min else 1
            for i, (xx, yy) in enumerate(coords):
                val = acts[i]
                norm_val = (val - a_min)/a_range
                color = plt.cm.gray(1.0 - norm_val/1.5)
                ax.plot(xx, yy, 'o', markerfacecolor=color, markeredgecolor='black',markersize=ms)
                ax.text(xx, yy, f"{val:.2f}", ha='center', va='center',
                        color='black', fontsize=fs)
                
        # (1) 입력 노드
        ax.plot(node_coords['input'][0][0], node_coords['input'][0][1],
                's', markerfacecolor='white', markeredgecolor='black', markersize=22)
        ax.text(node_coords['input'][0][0], node_coords['input'][0][1],
                "Input", ha='center', va='center',
                color='black', fontsize=8)

        # (2) hidden1
        plot_hidden(h1, node_coords['hidden1'], ms=20, fs=5)
        # (3) hidden2
        plot_hidden(h2, node_coords['hidden2'], ms=20, fs=5)
        # (4) hidden3
        plot_hidden(h3, node_coords['hidden3'], ms=20, fs=6)

        # (5) output (2차원)
        out_min, out_max = out.min(), out.max()
        out_range = (out_max - out_min) if out_max != out_min else 1
        for i, (xx, yy) in enumerate(node_coords['output']):
            val = out[i]
            norm_val = (val - out_min) / out_range
            color = plt.cm.gray(1 - norm_val/2)  # 활성화 값 높을수록 어두운 색
            ax.plot(xx, yy, 's', markerfacecolor=color, markeredgecolor='black',
                    markersize=22)  # 사각형 출력
            ax.text(xx, yy, f"{val:.2f}", ha='center', va='center', fontsize=8, color='black')
            ax.text(xx + 0.1, yy, f"Class {i+1}", ha='left', va='center', fontsize=8)


        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
