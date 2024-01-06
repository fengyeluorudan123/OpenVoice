import os
import torch
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter

ckpt_base = '/home/admin/workspace/aop_lab/app_source/OpenVoice/checkpoints/base_speakers/EN'
ckpt_converter = '/home/admin/workspace/aop_lab/app_source/OpenVoice/checkpoints/converter'
device = 'cuda:0'
output_dir = 'outputs'


class OpenVoiceInfer():
    def __init__(self):
        self.access_key_id = 'LTAI5tLiMuempJBF6vis3WWW'
        self.access_key_secret = 'zSRNWh9ypFPRUgSSdyt5ni5khUyt6K'
        self.endpoint = 'oss-cn-beijing.aliyuncs.com'
        self.bucket_name = 'fengyeluo'

    def forward(self, yinsepth, text, style):
        # 从oss下载这个文件的byte流到本地

        print(1)

        ckpt_base = '/root/OpenVoice/checkpoints/base_speakers/EN'
        # 基础模型
        ckpt_converter = '/root/OpenVoice/checkpoints/converter'
        device = 'cuda:0'
        output_dir = '/root/OpenVoice/outputs'

        # 获取基础的tts组件
        base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
        base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

        # 获取基础的tts组件
        tone_color_converter = ToneColorConverter('/root/OpenVoice/checkpoints/converter/config.json', device=device)
        tone_color_converter.watermark_model = None
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        os.makedirs(output_dir, exist_ok=True)

        # 记载原始的底模的模型问津
        source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
        # 给一段参考的mp3
        # target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)
        target_se = torch.load(yinsepth, map_location=device)

        save_path = f'{output_dir}/output_en_default.wav'
        # 用户输入文字以及用户名
        # text = "The error message you're encountering is indicating that there is a conflict between the dependencies of the packages you are trying to install. Specifically, it's related to the version of the package you're trying to install conflicting with the requirements of other packages"
        # text = "我的名字叫做洛天依，今年23岁了，今天是2024年一月一日，我祝大家新年快乐"
        src_path = f'{output_dir}/tmp.wav'
        # 基础声音大模型
        base_speaker_tts.tts(text, src_path, speaker='default', language='english', speed=1.0)

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convertToSpeech(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message)



