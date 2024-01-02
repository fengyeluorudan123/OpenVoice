import oss2
from io import BytesIO

class OSSUtil:
    def __init__(self, access_key_id, access_key_secret, endpoint, bucket_name):
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)

    def upload_bytes(self, data, object_name):
        try:
            self.bucket.put_object(object_name, data)
            return f"https://{self.bucket.bucket_name}.{self.bucket.endpoint}/{object_name}"
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

# 使用示例
if __name__ == "__main__":
    access_key_id = 'LTAI5tLiMuempJBF6vis3WWW'
    access_key_secret = 'zSRNWh9ypFPRUgSSdyt5ni5khUyt6K'
    endpoint = 'oss-cn-beijing.aliyuncs.com'
    bucket_name = 'oss-cn-beijing.aliyuncs.com'

    oss_util = OSSUtil(access_key_id, access_key_secret, endpoint, bucket_name)

    # 上传字节流到OSS
    data_to_upload = b'This is a test'
    uploaded_url = oss_util.upload_bytes(data_to_upload, 'test.txt')
    if uploaded_url:
        print(f"Uploaded URL: {uploaded_url}")

    # 根据链接获取文件字节流
    downloaded_data = oss_util.download_bytes('test.txt')
    if downloaded_data:
        print(f"Downloaded Data: {downloaded_data.getvalue()}")

