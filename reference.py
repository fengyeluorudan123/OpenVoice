import os
import torch
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter



#基础模型
ckpt_base = '/root/OpenVoice/checkpoints/base_speakers/EN'
#基础模型
ckpt_converter = '/root/OpenVoice/checkpoints/converter'
device = 'cuda:0'
output_dir = '/root/OpenVoice/outputs'

base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

tone_color_converter = ToneColorConverter('/root/OpenVoice/checkpoints/converter/config.json', device=device)
tone_color_converter.watermark_model = None
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

username = "temp"

source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
#给一段参考的mp3
#target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)
target_se = torch.load('/root/OpenVoice/processed/567/se.pth', map_location=device)


save_path = f'{output_dir}/output_en_default.wav'
# 用户输入文字以及用户名
text = "The error message you're encountering is indicating that there is a conflict between the dependencies of the packages you are trying to install. Specifically, it's related to the version of the package you're trying to install conflicting with the requirements of other packages"
#text = "我的名字叫做洛天依，今年23岁了，今天是2024年一月一日，我祝大家新年快乐"
src_path = f'{output_dir}/tmp.wav'
#基础声音大模型
base_speaker_tts.tts(text, src_path, speaker='default', language='english', speed=1.0)

# Run the tone color converter
encode_message = "@MyShell"
tone_color_converter.convertToSpeech(
    audio_src_path=src_path, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=save_path,
    message=encode_message)