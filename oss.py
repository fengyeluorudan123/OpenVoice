import oss2
from io import BytesIO
import tempfile

access_key_id = 'LTAI5tLiMuempJBF6vis3WWW'
access_key_secret = 'zSRNWh9ypFPRUgSSdyt5ni5khUyt6K'
endpoint = 'oss-cn-beijing.aliyuncs.com'
bucket_name = 'fengyeluo'


class OSSUtil:
    def __init__(self, access_key_id, access_key_secret, endpoint, bucket_name):
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)

    def upload_bytes(self, data, object_name, folder=None):
        try:
            if folder:
                # 如果提供了文件夹名，确保路径以'/'结束
                folder = folder.rstrip('/') + '/'
                object_name = folder + object_name
            self.bucket.put_object(object_name, data)
            print(self.bucket.endpoint)
            print(self.bucket.bucket_name)
            return f"https://{self.bucket.bucket_name}.{endpoint}/{object_name}"
        except Exception as e:
            print(f"Error uploading object: {e}")
            return None

    def download_bytes(self, object_name):
        try:
            obj = self.bucket.get_object(object_name)
            return BytesIO(obj.read())
        except Exception as e:
            print(f"Error downloading object: {e}")
            return None

    def download_to_temp_file(self, object_name):
        print(18)
        byte_stream = self.download_bytes(object_name)
        print(18)
        if byte_stream is not None:
            # 使用tempfile创建一个临时文件,需要修改为一个临时文件夹下的临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            # 将字节流的内容写入临时文件
            temp_file.write(byte_stream.getvalue())
            # 获取临时文件的路径
            temp_file_path = temp_file.name
            # 关闭文件句柄
            temp_file.close()
            # 返回临时文件的路径
            return temp_file_path
        else:
            return None


# 使用示例
if __name__ == "__main__":

    access_key_id = 'LTAI5tLiMuempJBF6vis3WWW'
    access_key_secret = 'zSRNWh9ypFPRUgSSdyt5ni5khUyt6K'
    endpoint = 'oss-cn-beijing.aliyuncs.com'
    bucket_name = 'fengyeluo'
    oss_util = OSSUtil(access_key_id, access_key_secret, endpoint, bucket_name)

    # 上传字节流到OSS的指定文件夹
    folder_name = 'openvoice/vedio'
    # 目标文件夹
    buffer = BytesIO()

    # 保存张量到BytesIO对象，而不是文件
    mp3_file_path = '/home/admin/workspace/aop_lab/app_source/OpenVoice/example_reference.mp3'

    # 以二进制模式打开MP3文件
    with open(mp3_file_path, 'rb') as file:
        mp3_bytes = file.read()

    # 获取字节流
    # gs_bytes = buffer.getvalue()
    # data_to_upload = b'This is a test'
    uploaded_url = oss_util.upload_bytes(mp3_bytes, 'example_reference.mp3', folder=folder_name)
    if uploaded_url:
        print(f"Uploaded to folder URL: {uploaded_url}")

    # 根据链接获取文件字节流
    # object_path_in_folder = f"{folder_name}/hy.mp3"  # OSS中的完整路径
    # print(object_path_in_folder)
    # downloaded_data = oss_util.download_bytes(object_path_in_folder)
    # if downloaded_data:
    #     print(f"Downloaded Data: {downloaded_data.getvalue()}")