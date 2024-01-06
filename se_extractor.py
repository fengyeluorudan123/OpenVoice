import os
import glob
import torch
from glob import glob
import numpy as np
from pydub import AudioSegment
from oss import OSSUtil
# from api import ToneColorConverter
from faster_whisper import WhisperModel
from whisper_timestamped.transcribe import get_audio_tensor, get_vad_segments

model_size = "medium"
# Run on GPU with FP16
model = None

endpoint = 'oss-cn-beijing.aliyuncs.com'
bucket_name = 'fengyeluo'


def split_audio_whisper(audio_path, target_dir='processed'):
    global model
    if model is None:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)

    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    target_folder = os.path.join(target_dir, audio_name)

    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments)

    # create directory
    os.makedirs(target_folder, exist_ok=True)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)

    # segments
    s_ind = 0
    start_time = None

    for k, w in enumerate(segments):
        # process with the time
        if k == 0:
            start_time = max(0, w.start)

        end_time = w.end

        # calculate confidence
        if len(w.words) > 0:
            confidence = sum([s.probability for s in w.words]) / len(w.words)
        else:
            confidence = 0.
        # clean text
        text = w.text.replace('...', '')

        # left 0.08s for each audios
        audio_seg = audio[int(start_time * 1000): min(max_len, int(end_time * 1000) + 80)]

        # segment file name
        fname = f"{audio_name}_seg{s_ind}.wav"

        # filter out the segment shorter than 1.5s and longer than 20s
        save = audio_seg.duration_seconds > 1.5 and \
               audio_seg.duration_seconds < 20. and \
               len(text) >= 2 and len(text) < 200

        if save:
            output_file = os.path.join(wavs_folder, fname)
            audio_seg.export(output_file, format='wav')

        if k < len(segments) - 1:
            start_time = max(0, segments[k + 1].start - 0.08)

        s_ind = s_ind + 1
    return wavs_folder


def split_audio_vad(audio_path, target_dir, split_seconds=10.0):
    print(audio_path)
    print(target_dir)
    SAMPLE_RATE = 16000
    audio_vad = get_audio_tensor(audio_path)
    print(audio_vad.type)
    segments = get_vad_segments(
        audio_vad,
        output_sample=True,
        min_speech_duration=0.1,
        min_silence_duration=1,
        method="auditok",
    )
    segments = [(seg["start"], seg["end"]) for seg in segments]
    segments = [(float(s) / SAMPLE_RATE, float(e) / SAMPLE_RATE) for s, e in segments]
    print(segments)
    audio_active = AudioSegment.silent(duration=0)

    audio = AudioSegment.from_file(audio_path)

    for start_time, end_time in segments:
        audio_active += audio[int(start_time * 1000): int(end_time * 1000)]

    audio_dur = audio_active.duration_seconds
    print(f'after vad: dur = {audio_dur}')
    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)
    start_time = 0.
    count = 0
    num_splits = int(np.round(audio_dur / split_seconds))
    assert num_splits > 0, 'input audio is too short'
    interval = audio_dur / num_splits

    for i in range(num_splits):
        end_time = min(start_time + interval, audio_dur)
        if i == num_splits - 1:
            end_time = audio_dur
        output_file = f"{wavs_folder}/{audio_name}_seg{count}.wav"
        audio_seg = audio_active[int(start_time * 1000): int(end_time * 1000)]
        audio_seg.export(output_file, format='wav')
        start_time = end_time
        count += 1
    print(wavs_folder)
    return wavs_folder


def get_se(audio_path, vc_model, target_dir='processed', vad=True):
    device = vc_model.device

    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    se_path = os.path.join(target_dir, audio_name, 'se.pth')

    if os.path.isfile(se_path):
        se = torch.load(se_path).to(device)
        return se, audio_name
    if os.path.isdir(audio_path):
        wavs_folder = audio_path
    elif vad:
        wavs_folder = split_audio_vad(audio_path, target_dir)
    else:
        wavs_folder = split_audio_whisper(audio_path, target_dir)

    audio_segs = glob(f'{wavs_folder}/*.wav')
    if len(audio_segs) == 0:
        raise NotImplementedError('No audio segments found!')

    return vc_model.extract_se(audio_segs, se_save_path=se_path), audio_name


def get_se(audio_path, vc_model, user_id, voice_name, target_dir='processed', vad=True):
    print(3)
    # device = vc_model.device

    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    # se_path = os.path.join(target_dir, audio_name, 'se.pth')获取声音在oss端的文件名称
    last_part = audio_path.split('/')[-1].split('.')[0]
    # 拼接用户+类型+声音文件的名称,作为这个用户的这个风格的唯一格式路径
    object_name = last_part + '_' + user_id + '_' + voice_name + '.pth'

    # 检查OSS中是否存在模型文件

    oss_object_url = f"https://{bucket_name}.{endpoint}/openvoice/yinse/{object_name}"
    print(oss_object_url)

    access_key_id = 'LTAI5tLiMuempJBF6vis3WWW'
    access_key_secret = 'zSRNWh9ypFPRUgSSdyt5ni5khUyt6K'
    # endpoint = 'oss-cn-beijing.aliyuncs.com'
    # bucket_name = 'fengyeluo'
    oss_util = OSSUtil(access_key_id, access_key_secret, endpoint, bucket_name)
    if oss_util.bucket.object_exists('openvoice/yinse/' + last_part + '_' + user_id + '_' + voice_name + '.pth'):
        # 如果文件存在，直接返回OSS的地址
        print('该模型已经存在')
        return oss_object_url

    # 将远程的音色下载到本地的临时文件，均存在一个文件夹下面并且使用用户+音色区分开
    folder_name = '~/openvoice/vedio/temp/' + user_id + '/' + voice_name + '/'
    object_path_in_folder = folder_name + last_part + '_' + user_id + '_' + voice_name + ".mp3"
    print('download_to_temp_file' + audio_path)
    url = oss_util.download_to_temp_file(audio_path)
    print('url' + url)
    # if os.path.isfile(se_path):
    #     se = torch.load(se_path).to(device)
    #     return se, audio_name
    # if os.path.isdir(audio_path):
    #     wavs_folder = audio_path
    # elif vad:
    #     wavs_folder = split_audio_vad(audio_path, target_dir)
    # else:
    #     wavs_folder = split_audio_whisper(audio_path, target_dir)

    print(4)

    wavs_folder = split_audio_vad(url, folder_name)
    print(5)

    audio_segs = glob(f'{wavs_folder}/*.wav')
    if len(audio_segs) == 0:
        raise NotImplementedError('No audio segments found!')

    return vc_model.extract_se(audio_segs, se_save_path=object_name), audio_name


if __name__ == "__main__":
    ckpt_base = 'checkpoints/base_speakers/EN'
    ckpt_converter = 'checkpoints/converter'
    device = 'cuda:0'
    output_dir = 'outputs'

    # 加载本地的配置文件
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    # 加载本地的基础音色模型
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    reference_speaker = 'resources/example_reference.mp3'
    target_se, audio_name = get_se(reference_speaker,
                                   tone_color_converter,
                                   user_id="1000001",
                                   voice_name="test",
                                   target_dir='processed',
                                   vad=True,
                                   )