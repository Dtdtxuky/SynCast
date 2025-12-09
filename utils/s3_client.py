import os
import boto3
import botocore
from boto3.session import Session
from botocore.exceptions import ClientError
import logging
import time
import sys
import socket
import numpy as np
from io import BytesIO
import xarray as xr


# http://10.140.2.204
class s3_client():
    def __init__(self, bucket_name='myBucket', endpoint='http://10.140.31.254:80', user='zhangwenlong', jiqun = 's'):
        self.bucket_name = bucket_name

        if user == 'zhangwenlong' and jiqun== 's':
            # access_key = 'KHDS9L23TZRS0KPE38N4'
            # secret_key = 'rzg2BXyYi21Kx25EKKyOr0JkGm0ZHDBKgKWxk6BG'
            access_key = 'OJL0B2YGGU37OEOWD8TX'
            secret_key = '7kRVIBGlnFW2LW3XUBEW35HkXPPb4UYDQbcaS35K'
            endpoint = endpoint
        
        elif jiqun=='p':
            access_key = 'JFNE7R0ZLH84E6GAPLLQ'
            secret_key = '9hlz7zM53qVNsOCxUO1TofUXsOL4boR7O89ChOkH'
            endpoint = endpoint
            
        else:
            raise ValueError('user not defined')
    
            
        # endpoint = endpoint# url = "http://172.xx.xx.xxx"  # 也可以是自己节点的地址
        # ip = socket.gethostbyname(socket.gethostname())
        # import pdb; pdb.set_trace()
        # ip_prefix = ip.split('.')[-2]
        # if ip_prefix == '54' or ip_prefix == '52':
        #     endpoint = 'http://10.140.2.204' # outside ip
        # elif ip_prefix == '37':
        #     endpoint = 'http://10.140.3.253' # outside ip
        # else:
        #     endpoint = 'http://10.140.14.204'# url = "http://172.xx.xx.xxx"  # 也可以是自己节点的地址
        client_config = botocore.config.Config(
            max_pool_connections=1000,
        )

        client_config = botocore.config.Config(
            max_pool_connections=10000,
        )
        KB = 1024
        MB = KB * KB
        # self.trans_cfg = boto3.s3.transfer.TransferConfig(
        #     multipart_threshold=8 * MB,
        #     max_concurrency=20,
        #     multipart_chunksize=16 * MB,
        #     num_download_attempts=10,
        #     max_io_queue=100,
        #     io_chunksize=256 * MB,
        #     use_threads=True,
        #     max_bandwidth=30*MB*MB,
        # )

        session = Session(access_key, secret_key)
        self.s3_client = session.client('s3', endpoint_url=endpoint, config=client_config)

    def download_file(self, objectName, fileName,):
        """
        下载文件
        :param bucketName: 桶的名称
        :param objectName: 文件的路径
        :param fileName: 下载完成的文件的名称----注意：下载之后的文件默认储存在自己的python工程路径下
        :return:
        """
        self.s3_client.download_file(self.bucket_name,objectName,fileName)#,Config=self.trans_cfg)
        
    def read_grib_from_BytesIO(self, objectName):
        file_stream = BytesIO()
        self.s3_client.download_fileobj(self.bucket_name, objectName, file_stream)
        return  xr.open_dataset(file_stream, engine='cfgrib', backend_kwargs=dict(indexpath=None))
    
    def read_nc_from_BytesIO(self, objectName):
        file_stream = BytesIO()
        self.s3_client.download_fileobj(self.bucket_name, objectName, file_stream)
        return  xr.open_dataset(file_stream, engine='h5netcdf')
    
    def read_npy_from_BytesIO(self, objectName, bucket_name=None):
        # if self.get_filesize(self.bucket_name, objectName)==0:
        #     print(f'==================object {objectName} is not exist!!!==================')
        file_stream = BytesIO()
        if bucket_name:
            self.s3_client.download_fileobj(bucket_name, objectName, file_stream)
        else:
            self.s3_client.download_fileobj(self.bucket_name, objectName, file_stream)
        file_stream.seek(0)
        return  np.load(file_stream, allow_pickle=True)
    
    def upload_npy(self, data: np.array, bucket: str, s3_uri: str):
        # s3_uri looks like f"s3://{BUCKET_NAME}/{KEY}"
        bytes_ = BytesIO()
        np.save(bytes_, data, allow_pickle=True)
        bytes_.seek(0)
        try:
            self.s3_client.upload_fileobj(
                Fileobj=bytes_, Bucket=bucket, Key=s3_uri
            )
            return True
        except Exception as e:
            print(e)
            return False
    
    def upload_file(self, file_name, bucket_name=None, object_name=None):
        """
        上传文件
        :param file_name: 需要上传的文件的名称
        :param bucket: S3中桶的名称
        :param object_name: 需要上传到的路径，例如file/localfile/test
        :return:
        """
        st= time.time()
        file_size = os.path.getsize(file_name)
        if object_name is None:
            object_name = file_name
        self.s3_client.upload_file(file_name, bucket_name, object_name, ExtraArgs = { 'ACL' : 'public-read' } )#,Config=self.trans_cfg)
        speed = (1 * file_size) / (time.time() - st)
        speed_str = " Speed: %s" % self.format_size(speed)
        print(f"the upload speed is {speed_str}")
        # try:
        #     self.s3_client.upload_file(file_name, bucket_name, object_name, ExtraArgs = { 'ACL' : 'public-read' } )#,Config=self.trans_cfg)
        #     speed = (1 * file_size) / (time.time() - st)
        #     speed_str = " Speed: %s" % self.format_size(speed)
        #     print(f"the upload speed is {speed_str}")

        # except ClientError as e:
        #     logging.error(e)
        #     return False

        return True

    def get_filesize(self,bucketName=None, objectName=None):
        """
        下载文件
        :param bucketName: 桶的名称
        :param objectName: 文件的路径
        :param fileName: 下载完成的文件的名称----注意：下载之后的文件默认储存在自己的python工程路径下
        :return:
        """
        # import pdb
        # pdb.set_trace()
        file_list = self.list_object(self.bucket_name,prefix=objectName.split(',')[0])

        if  objectName in file_list:
            size = self.s3_client.head_object(Bucket=self.bucket_name, Key=objectName).get('ContentLength')
            return size
        else:
            return 0
        
    def list_object(self, bucketName, prefix):
        """
        列出当前桶下所有的文件
        :param bucketName:
        :return:
        """
        file_list = []
        response = self.s3_client.list_objects_v2(
            Bucket=bucketName,
            Prefix=prefix,
            MaxKeys=1000  # 返回数量，如果为空则为全部
        )

        if 'Contents' not in response.keys():
            return  []

        else:
            file_desc = response['Contents']
            for f in file_desc:
                # print('file_name:{},file_size:{}'.format(f['Key'], f['Size']))
                file_list.append(f['Key'])
            # import pdb
            # pdb.set_trace()
            return file_list

    def Schedule(blocknum, blocksize, totalsize):
        speed = (blocknum * blocksize) / (time.time() - start_time)
        # speed_str = " Speed: %.2f" % speed
        speed_str = " Speed: %s" % self.format_size(speed)
        recv_size = blocknum * blocksize

        # 设置下载进度条
        f = sys.stdout
        pervent = recv_size / totalsize
        percent_str = "%.2f%%" % (pervent * 100)
        n = round(pervent * 50)
        s = ('#' * n).ljust(50, '-')
        f.write(percent_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
        f.flush()
        # time.sleep(0.1)
        f.write('\r')

        # 字节bytes转化K\M\G
        
    def format_size(self,bytes):
        try:
            bytes = float(bytes)
            kb = bytes / 1024
        except:
            print("传入的字节格式不对")
            return "Error"
        if kb >= 1024:
            M = kb / 1024
            if M >= 1024:
                G = M / 1024
                return "%.3fG" % (G)
            else:
                return "%.3fM" % (M)
        else:
            return "%.3fK" % (kb)
