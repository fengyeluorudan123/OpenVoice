import os
import torch
import se_extractor
from oss import OSSUtil
from io import BytesIO
from api import BaseSpeakerTTS, ToneColorConverter

ckpt_base = '/home/admin/workspace/aop_lab/app_source/OpenVoice/checkpoints/base_speakers/EN'
ckpt_converter = '/home/admin/workspace/aop_lab/app_source/OpenVoice/checkpoints/converter'
device = 'cuda:0'
output_dir = 'outputs'

class OpenVoiceTrainer():
    def __init__(self):
        self.access_key_id = 'LTAI5tLiMuempJBF6vis3WWW'
        self.access_key_secret = 'zSRNWh9ypFPRUgSSdyt5ni5khUyt6K'
        self.endpoint = 'oss-cn-beijing.aliyuncs.com'
        self.bucket_name = 'fengyeluo'



    def forward(self, sourceurl, userId, style):

        #从oss下载这个文件的byte流
        print(1)


        # 加载本地的配置文件
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        print(2)
        # 加载本地的基础音色模型
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        print(2)

        # 接受外部输入的音色文件
        target_se = se_extractor.get_se(sourceurl,
                                                    tone_color_converter,
                                                    userId,
                                                    style,
                                                    target_dir='processed',
                                                    vad=True,
                                                    )




        return target_se


#上传返回链接
if __name__ == "__main__":
    voice = OpenVoiceTrainer()
    voice.forward('openvoice/vedio/hu.mp3','100000','testhy')



    # access_key_id = 'LTAI5tLiMuempJBF6vis3WWW'
    # access_key_secret = 'zSRNWh9ypFPRUgSSdyt5ni5khUyt6K'
    # endpoint = 'oss-cn-beijing.aliyuncs.com'
    # bucket_name = 'fengyeluo'
    # oss_util = OSSUtil(access_key_id, access_key_secret, endpoint, bucket_name)
    #
    # # 上传字节流到OSS的指定文件夹
    # folder_name = 'openvoice/vedio'
    # object_path_in_folder = f"{folder_name}/hy.mp3"
    # # 目标文件夹
    # buffer = BytesIO()
    #
    # # 保存张量到BytesIO对象，而不是文件
    # mp3_file_path = 'hy.mp3'
    #
    # # 以二进制模式打开MP3文件
    # with open(mp3_file_path, 'rb') as file:
    #     mp3_bytes = file.read()
    #
    # # 获取字节流
    # #gs_bytes = buffer.getvalue()
    # #data_to_upload = b'This is a test'
    # uploaded_url = oss_util.upload_bytes(mp3_bytes, 'hy.mp3', folder=folder_name)
    # print(uploaded_url)
    # if uploaded_url:
    #     print(f"Uploaded to folder URL: {uploaded_url}")
    #
    # # 根据链接获取文件字节流
    # object_path_in_folder = f"{folder_name}/hy.mp3"  # OSS中的完整路径
    # print(object_path_in_folder)
    # downloaded_data = oss_util.download_bytes(object_path_in_folder)
    # if downloaded_data:
    #     print(f"Downloaded Data: {downloaded_data.getvalue()}")
    #
    # #将用户和类型以及声音的流文件进行音色的训练
