import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import os
import glob

cred = credentials.Certificate('testuploadfile-9d666-firebase-adminsdk-iomvc-28668e19a4.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'testuploadfile-9d666.appspot.com'
})

def upload_image(file_path, destination_path):
    bucket = storage.bucket()
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(file_path)

def download_image(source_path, destination_path):
    bucket = storage.bucket()
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)

def list_all_items_in_firebase():
    bucket = storage.bucket()
    blobs = bucket.list_blobs()

    for blob in blobs:
        print(blob.name)

def list_all_items_in_host(directory):
        # Tạo đường dẫn đầy đủ của thư mục
    directory_path = os.path.abspath(directory)

    # Sử dụng glob để lấy danh sách tệp ảnh trong thư mục
    image_files = glob.glob(os.path.join(directory_path, '*.jpg'))  # Có thể thay '*.jpg' thành phần mở rộng tệp ảnh khác

    # Lấy tên tệp ảnh từ đường dẫn đầy đủ
    image_filenames = [os.path.basename(file) for file in image_files]

    return image_filenames