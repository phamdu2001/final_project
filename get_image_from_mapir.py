import requests
from bs4 import BeautifulSoup

camera_ip = '192.168.0.100'

camera_url = f'http://{camera_ip}/'

response = requests.get(camera_url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    image_tags = soup.find_all('img')

    for img_tag in image_tags:
        image_src = img_tag['src']
        
        image_response = requests.get(camera_url + image_src)
        if image_response.status_code == 200:
            with open(f'image_{image_src}.jpg', 'wb') as image_file:
                image_file.write(image_response.content)
                print(f'Lưu ảnh {image_src} thành công.')
        else:
            print(f'Lỗi khi tải xuống ảnh {image_src}.')
else:
    print('Lỗi khi truy cập vào giao diện web của camera.')