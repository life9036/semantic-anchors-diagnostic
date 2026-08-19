# =============================================================================
# demo/t2v.py —— 文本到视频（Text-to-Video）演示脚本
# =============================================================================
# 这个文件做了什么？
#   输入一句英文提示词（prompt），输出一个 mp4 视频，里面是一个 stickman
#   （火柴人）在做提示词里描述的动作。
#
# 整体数据流（非常重要，先看一遍再读代码）：
#
#   "walk fast"   ──►  ① 文本编码器  ──►  512维向量
#                                         │
#                                         ▼
#                          ② 想象模块（connector + RSSM）
#                              把"一句话"伪装成"一段视频的特征序列"
#                              让世界模型在隐空间里"想象"接下来会发生什么
#                                         │
#                                         ▼
#                          ③ 像素解码器（CNN decoder）
#                              把隐状态还原成 64×64 RGB 帧
#                                         │
#                                         ▼
#                          ④ imageio 写 mp4 文件
#
# 关键概念：
#   - 文本/视频特征共享同一个 512 维空间
#     （因为 InternVideo2 在训练时把图文嵌进了同一空间）
#   - "connector" 是一个去噪自编码器（denoising AE），
#     它把"文本特征"当作"被噪声污染的视频特征"，
#     训练时学的是"去噪还原"。推理时用同一套去噪机制把文本特征
#     拉回到世界模型见过的"视频特征"分布里。
#   - 世界模型（Dreamer RSSM）只需要 512 维向量作为输入，
#     不在乎这个向量到底来自文本还是视频。
#
# 这个文件不参与训练，只做推理（inference）演示。
# 训练逻辑在 agent/、tools/、envs/ 下面。
# =============================================================================


# ---------- 标准库与第三方 import ----------
from pathlib import Path
import os
import sys

# 把当前工作目录加进 sys.path，这样下面的 `from tools.xxx` 才能找到模块。
# 注意：Path(os.path.abspath('')) 是"当前命令行所在目录"，
#       不是 t2v.py 所在目录。所以必须在仓库根目录 /root/autodl-tmp/genrl 下运行。
sys.path.append(str(Path(os.path.abspath(''))))

import torch                      # 深度学习框架
import numpy as np                # 数值计算（虽然本文件实际没用到，但保留以备扩展）
from tools.genrl_utils import ViCLIPGlobalInstance  # InternVideo2 的加载器

import time                       # 仅用于打时间戳
import torchvision                # 仅用于 make_grid（多帧拼图）
from huggingface_hub import hf_hub_download  # 用于自动从 HuggingFace 下载权重


# =============================================================================
# save_videos —— 把一个张量序列写成一个 mp4 文件
# =============================================================================
# 输入张量形状：(B, samples, C, T, H, W)
#   B         = batch size（一次生成几条视频）
#   samples   = 同一个 prompt 想要的几条"候选"视频（这里只取 1）
#   C, T, H, W= 通道数(3)、时间帧数、高、宽
# 输出：savedir/<filename>.mp4
def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b, samples, c, t, h, w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        # 拿到 batch 中第 idx 个候选视频
        video = vid_tensor.detach().cpu()
        # 把像素值限制在 [0, 1]（解码器可能输出略微越界）
        video = torch.clamp(video.float(), 0., 1.)
        # 维度重排：把 (samples, c, t, h, w) → (t, samples, c, h, w)
        # 之后按时间维遍历，每一帧把"多条候选"横向拼起来
        video = video.permute(1, 0, 2, 3, 4)  # t, n, c, h, w
        # torchvision.utils.make_grid 会把 n 张图横拼成一张大图
        # 这里每帧的"候选图"都拼起来，方便整段视频一起看
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples))
                       for framesheet in video]   # 每一帧 → [3, H, n*W]
        # 把所有帧沿时间维堆叠：得到 (T, 3, H, W*n)
        grid = torch.stack(frame_grids, dim=0)
        # 转换到 uint8 + 把通道放最后（imageio 期望 HWC）
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        # 用 imageio-ffmpeg 写 mp4：
        #   - libx264 是常见的 H.264 编码器
        #   - macro_block_size=1 避免非 16 像素倍数的尺寸被拒绝
        import imageio
        writer = imageio.get_writer(savepath, fps=fps, codec='libx264',
                                    quality=8, macro_block_size=1)
        for frame in grid.numpy():
            writer.append_data(frame)
        writer.close()


# =============================================================================
# Text2Video —— 文本生成视频的核心类
# =============================================================================
# 在 __init__ 中：
#   1) 下载/加载 GenRL 的世界模型权重（genrl_stickman_500k_2.pt）
#   2) 下载/加载 InternVideo2 权重（提供文本编码能力）
#   3) 把两个模型都放到 GPU 上
#
# 在 get_prompt() 中：
#   给一句 prompt + duration，输出一段 mp4
class Text2Video():
    def __init__(self, result_dir='./tmp/', gpu_num=1) -> None:
        # ------- 第 1 步：定位/下载 GenRL 自家模型 -------
        # genrl_stickman_500k_2.pt 是论文训练好的世界模型（含 RSSM + connector + decoder）
        model_folder = str(Path(os.path.abspath('')) / 'models')
        model_filename = 'genrl_stickman_500k_2.pt'

        # 模型不存在就自动从 HuggingFace 下载（REPO = mazpie/genrl_models）
        if not os.path.isfile(os.path.join(model_folder, model_filename)):
            self.download_model(model_folder, model_filename)
        # InternVideo2 也是同样套路（用作"文本/视频编码器"）
        if not os.path.isfile(os.path.join(model_folder, 'InternVideo2-stage2_1b-224p-f4.pt')):
            self.download_internvideo2(model_folder)

        # ------- 第 2 步：把 GenRL 模型加载进内存 -------
        # 注意：torch.load 默认 weights_only=True，PyTorch ≥2.6 会拒绝加载自定义对象。
        #       当前脚本没加 weights_only=False，运行时会报错，
        #       需要手动 patch（见 README / 我们之前的对话）。
        self.agent = torch.load(os.path.join(model_folder, model_filename))
        model_name = 'internvideo2'

        # ------- 第 3 步：实例化 InternVideo2（提供文本编码能力） -------
        # ViCLIPGlobalInstance 是项目自己写的 InternVideo2 包装类
        # 它会做：import 第三方代码 → 读 config → 加载权重 → 放到 cuda → 设 n_frames=8
        viclip_global_instance = ViCLIPGlobalInstance(model_name)
        if not viclip_global_instance._instantiated:
            print("Instantiating InternVideo2")
            viclip_global_instance.instantiate()
        # self.clip        ：InternVideo2 模型本身
        # self.tokenizer   ：对应的 tokenizer（这个 demo 里其实没用上）
        self.clip = viclip_global_instance.viclip
        self.tokenizer = viclip_global_instance.viclip_tokenizer

        # ------- 第 4 步：准备输出目录 -------
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    # -------------------------------------------------------------------------
    # get_prompt —— 真正的"一句话→视频"函数
    # -------------------------------------------------------------------------
    # prompt   : 英文文本，例如 "walk fast"
    # duration : 想象的时间步数（不是秒！越大视频越长，但画质保真度会下降）
    # 返回     : 写出的 mp4 的绝对路径
    def get_prompt(self, prompt, duration):
        # 每次调用先清一下显存（避免前面残留的 cache 干扰）
        torch.cuda.empty_cache()
        print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        start = time.time()

        # 文件名清洗：把 "/" 替换成 "_slash_"（避免子目录），空格替换成 "_"
        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str

        labels_list = [prompt_str]

        # 整个生成过程不计算梯度（纯推理）
        with torch.no_grad():
            # 取出 GenRL 世界模型的几个关键子模块
            wm = world_model = self.agent.wm                  # 整个世界模型
            connector = self.agent.wm.connector               # 视频-动作连接器（含去噪 AE）
            decoder = world_model.heads['decoder']            # 把隐状态解码回像素的 CNN
            n_frames = connector.n_frames                     # = 8，训练时的 chunk 长度

            # ============== ① 文本 → 512 维向量 ==============
            text_feat = []
            for text in labels_list:
                with torch.no_grad():
                    # InternVideo2 的文本编码器，把字符串映射到 512 维向量
                    text_feat.append(self.clip.get_txt_feat(text,))
            # 堆叠成 (B=1, 512)
            text_feat = torch.stack(text_feat, dim=0).to(self.clip.device)

            # 关键命名：把"文本特征"重新命名成"视频特征"。
            # 从世界模型的角度看，它根本不区分输入是文本还是视频，
            # 都当作"动作信号"即可。
            video_embed = text_feat

            # 下面 B / T / n_frames 是从训练代码里沿袭下来的变量名，
            # 在推理时并不直接被引用；保留只是为了让代码看起来与训练侧对齐。
            B = video_embed.shape[0]   # batch size，这里固定为 1
            T = 1                     # 初始时间步（之后会被 repeat 展开）

            # ============== ② 沿时间维重复 duration 次 ==============
            # 形状从 (B, 1, 512) → (B, duration, 512)
            # 含义：假装我们有一整段视频的特征，每一帧都是同一个 prompt 的"内容"
            video_embed = video_embed.repeat(1, duration, 1)

            # ============== ③ 想象 + 解码 ==============
            with torch.no_grad():
                # 在世界模型的隐空间里"想象"接下来会发生什么
                # sample=False         → 取 mean（更确定、更"平均"的动作）
                # reset_every_n_frames=False → 连续推演，不在 chunk 边界重置
                # denoise=True         → 关键！让 connector 内部的去噪自编码器
                #                          把"文本特征"当作"被污染的视频特征"
                #                          还原成世界模型认识的有效特征
                prior = wm.connector.video_imagine(
                    video_embed, None,
                    sample=False,
                    reset_every_n_frames=False,
                    denoise=True,
                )
                # prior 是一组隐状态字典（deter / stoch 等）
                # decoder_input_fn 把它们拼成 decoder 能吃的输入向量
                # decoder 把它解码成像素
                # .mean 取分布的均值（mode）
                # + 0.5 是因为 GenRL 在训练时把像素归一化到 [-0.5, 0.5]，
                #   加 0.5 才能得到 [0, 1] 的可视范围
                prior_recon = decoder(wm.decoder_input_fn(prior))['observation'].mean + 0.5

        # ============== ④ 写 mp4 文件 ==============
        # unsqueeze(0) 是给 save_videos 一个 batch 维 (B=1, samples=1, ...)
        save_videos(prior_recon.unsqueeze(0), self.result_dir,
                    filenames=[prompt_str], fps=15)
        print(f"Saved in {prompt_str}.mp4. Time used: {(time.time() - start):.2f} seconds")
        return os.path.join(self.result_dir, f"{prompt_str}.mp4")

    # -------------------------------------------------------------------------
    # download_model —— 下载 GenRL 自家模型（世界模型权重）
    # -------------------------------------------------------------------------
    def download_model(self, model_folder, model_filename):
        REPO_ID = 'mazpie/genrl_models'                 # HuggingFace 仓库 ID
        filename_list = [model_filename]
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        for filename in filename_list:
            local_file = os.path.join(model_folder, filename)
            if not os.path.exists(local_file):
                # 从 HuggingFace 仓库下载指定文件到本地
                hf_hub_download(repo_id=REPO_ID, filename=filename,
                                local_dir=model_folder, local_dir_use_symlinks=False)

    # -------------------------------------------------------------------------
    # download_internvideo2 —— 下载 InternVideo2（用作文本/视频编码器）
    # -------------------------------------------------------------------------
    def download_internvideo2(self, model_folder):
        REPO_ID = 'OpenGVLab/InternVideo2-Stage2_1B-224p-f4'
        filename_list = ['InternVideo2-stage2_1b-224p-f4.pt']
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        for filename in filename_list:
            local_file = os.path.join(model_folder, filename)
            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename,
                                local_dir=model_folder, local_dir_use_symlinks=False)


if __name__ == '__main__':
    t2v = Text2Video()
    video_path = t2v.get_prompt('doing a backflip', 64)
    print('done', video_path)
