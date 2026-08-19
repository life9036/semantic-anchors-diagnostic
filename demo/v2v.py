# =============================================================================
# demo/v2v.py —— 视频到视频（Video-to-Video）演示脚本
# =============================================================================
# 这个文件做了什么？
#   输入一段 mp4 示范视频（例如你在摄像头前做一个"走路"动作），
#   输出一个 mp4，里面是一个 stickman（火柴人）做相同动作的视频。
#
# 整体数据流（与 t2v.py 对比来理解）：
#
#   demo_video.mp4
#       │
#       │ ① 抽 8 帧
#       ▼
#   [T, 3, 224, 224] 视频帧
#       │
#       │ ② InternVideo2 视频编码器
#       ▼
#   512 维向量序列
#       │
#       │ ③ 想象模块（connector + RSSM）
#       ▼
#   隐状态
#       │
#       │ ④ CNN decoder
#       ▼
#   64×64 RGB 帧序列
#       │
#       ▼
#   out.mp4
#
# 与 t2v.py 的核心区别：
#   - t2v.py 用 `clip.get_txt_feat(text)` 拿到文本特征
#   - v2v.py 用 `clip.get_vid_features(frames)` 拿到视频特征
#   - 两者都喂给同一个 `wm.connector.video_imagine(...)`
#   - 因为 InternVideo2 把图文嵌进同一空间，模型根本不区分输入是文本还是视频
# =============================================================================


# ---------- 标准库与第三方 import ----------
from pathlib import Path
import os
import sys
import math

# 同 t2v.py：必须从仓库根目录运行，否则 `from tools.xxx` 找不到
sys.path.append(str(Path(os.path.abspath(''))))

import torch
import numpy as np
import cv2                                  # 读 mp4 抽帧
import imageio                              # 写 mp4
import torchvision                          # 用于 make_grid

# 复用 t2v.py 的 save_videos（写 mp4 的工具）
from demo.t2v import save_videos
from tools.genrl_utils import ViCLIPGlobalInstance


# =============================================================================
# 通用工具
# =============================================================================

def frames2tensor(frames, device):
    """
    把一段 numpy 帧序列 [T, H, W, C] (uint8) 转成 InternVideo2 期待的输入
        [1, T, 3, 224, 224] (float32, 归一化)

    参数
    ----
    frames   : list[np.ndarray] 或 np.ndarray，shape (T, H, W, 3)，uint8
    device   : torch.device
    """
    # InternVideo2 用 ImageNet 均值方差做归一化
    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    arr = np.stack(frames, axis=0).astype(np.float32)  # (T, H, W, 3)
    # cv2.resize 只能吃单张图 (H, W, 3)，必须逐帧 resize
    resized = np.stack([cv2.resize(f, (224, 224)) for f in arr], axis=0)  # (T, 224, 224, 3)
    normed = (resized / 255.0 - v_mean) / v_std          # (T, 224, 224, 3)
    # (T, H, W, C) → (T, C, H, W)
    normed = normed.transpose(0, 3, 1, 2)
    # → (1, T, C, H, W) 并转 tensor
    t = torch.from_numpy(normed).unsqueeze(0).to(device, non_blocking=True).float()
    return t


def load_video_clip(video_path, n_frames=8):
    """
    从本地 mp4 中均匀抽 n_frames 帧，返回 list[np.ndarray] (H, W, 3) uint8。

    抽帧策略：在视频总时长内等距取 n_frames 个时间点。
    如果视频帧数 < n_frames，则循环填充到 n_frames（notebook 里的做法）。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # 退路：逐帧读
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    else:
        # 等距抽帧
        indices = np.linspace(0, total - 1, n_frames).astype(int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                break
            # cv2 默认 BGR，需要转 RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    if len(frames) < n_frames:
        # 不够 8 帧就循环补
        while len(frames) < n_frames:
            frames.append(frames[-1])

    return frames[:n_frames]


# =============================================================================
# Video2Video 主类
# =============================================================================
class Video2Video:
    def __init__(self, result_dir='./tmp/') -> None:
        # ------- 加载 GenRL 世界模型 -------
        model_folder = str(Path(os.path.abspath('')) / 'models')
        model_filename = 'genrl_stickman_500k_2.pt'
        if not os.path.isfile(os.path.join(model_folder, model_filename)):
            raise FileNotFoundError(
                f"找不到 {model_filename}，请先跑 demo/t2v.py 触发自动下载，"
                f"或手动放到 {model_folder}/ 下"
            )
        # 注意：PyTorch ≥2.6 需 weights_only=False
        self.agent = torch.load(
            os.path.join(model_folder, model_filename),
            weights_only=False,
        )

        # ------- 加载 InternVideo2 -------
        viclip = ViCLIPGlobalInstance('internvideo2')
        if not viclip._instantiated:
            viclip.instantiate()
        self.clip = viclip.viclip

        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # get_demo —— "一段示范视频 → 一段 stickman 视频"
    # -------------------------------------------------------------------------
    # video_path : 示范视频路径（mp4/avi/mov 都行，只要 cv2 能读）
    # duration   : 想生成多长的视频（沿时间维 repeat 的次数，不是秒数）
    # filename   : 输出 mp4 的文件名（不含后缀）
    def get_demo(self, video_path, duration=16, filename=None):
        torch.cuda.empty_cache()

        if filename is None:
            filename = Path(video_path).stem

        # ===== ① 读示范视频，抽 8 帧 =====
        frames = load_video_clip(video_path, n_frames=self.clip.n_frames)
        # 拼成 InternVideo2 想要的输入
        vid_tensor = frames2tensor(frames, device=self.clip.device)  # (1, 8, 3, 224, 224)

        # ===== ② InternVideo2 视频塔 → 512 维向量 =====
        with torch.no_grad():
            # 取视频特征：(1, 8, 3, 224, 224) → (1, 512)
            vid_feat = self.clip.get_vid_features(vid_tensor)
        print(f"示范视频特征形状：{vid_feat.shape}")

        # ===== ③ 取出世界模型子模块 =====
        wm = self.agent.wm
        decoder = wm.heads['decoder']
        connector = wm.connector

        # ===== ④ 把特征"伪装"成动作序列 =====
        # 形状 (1, 1, 512) → (1, duration, 512)
        # 这一步与 t2v.py 完全一样：把单一特征沿时间维复制，
        # 让世界模型认为这是一段"连续视频"的动作信号
        video_embed = vid_feat.unsqueeze(1).repeat(1, duration, 1)

        # ===== ⑤ 世界模型想象 + 解码 =====
        with torch.no_grad():
            prior = wm.connector.video_imagine(
                video_embed, None,
                sample=False,
                reset_every_n_frames=False,
                denoise=True,   # 关键：让 connector 把"视频特征"对齐到世界模型分布
            )
            prior_recon = decoder(wm.decoder_input_fn(prior))['observation'].mean + 0.5

        # ===== ⑥ 写 mp4 =====
        save_videos(
            prior_recon.unsqueeze(0),           # (B=1, samples=1, C, T, H, W)
            self.result_dir,
            filenames=[filename],
            fps=15,
        )
        out_path = os.path.join(self.result_dir, f"{filename}.mp4")
        print(f"Saved: {out_path}")
        return out_path


# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Video-to-Video: 示范视频 → stickman 视频")
    parser.add_argument('video', type=str, help='示范视频路径（mp4/avi/mov）')
    parser.add_argument('--duration', type=int, default=16, help='生成视频长度（时间步数）')
    parser.add_argument('--out', type=str, default=None, help='输出文件名（不含后缀）')
    args = parser.parse_args()

    v2v = Video2Video()
    out = v2v.get_demo(args.video, duration=args.duration, filename=args.out)
    print('done', out)
