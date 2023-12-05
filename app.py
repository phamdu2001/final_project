import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import date
import base64
import os

current_folder_path = os.getcwd()
current_folder_path = current_folder_path.replace("\\", "/")
image = None
file_path = None
result = None
name = None

cred = credentials.Certificate(current_folder_path + '/tomato-6f924-firebase-adminsdk-5mh3b-1382727a0b.json')
firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://tomato-6f924-default-rtdb.asia-southeast1.firebasedatabase.app/'
})      

def normal(image, a=1/100, b = 15):
    # Chuyển đổi hình ảnh thành ma trận
    image_matrix = np.array(image)
    # hist, bins = np.histogram(image.flatten(), bins=256, range=[20, 120])
    mean = int(np.mean(image_matrix))

    # Áp dụng hàm bậc 2 vào từng phần tử trong ma trận
    rows, cols = image_matrix.shape
    for i in range(rows):
        for j in range(cols):
            if image_matrix[i][j] < 5:
                continue
            if image_matrix[i][j] > 230:
                continue
            
            image_matrix[i][j] = image_matrix[i][j] + (image_matrix[i][j] - mean) * ( a * mean ) + b
            
            if image_matrix[i][j] > 255:
                image_matrix[i][j] = 255

    return image_matrix
def RGN2NVDI_green(image, cl = 'r'):
    red_band = image[:, :, 1].astype(float)
    nir_band = image[:, :, 2].astype(float)
    if(cl == 'r'): 
        b = 0
        a = 1/100
    elif (cl == 'y'):
        b = 5
        a = 1/300
    else:
        a = 1/100
        b = 15
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi_normalized = (ndvi + 1) * 127.5
    ndvi_normalized = ndvi_normalized.astype(np.uint8)
    ndvi_normalized = normal(ndvi_normalized,a,b)
    return ndvi_normalized
def RGN2NVDI(image, cl = 'r'):
    if(cl == 'r'): 
        b = 0
        a = 1/100
    elif (cl == 'y'):
        b = 15
        a = 1/300
    else:
        a = 1/100
        b = 15
    red_band = image[:, :, 0].astype(float)
    nir_band = image[:, :, 2].astype(float)
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi_normalized = (ndvi + 1) * 127.5
    ndvi_normalized = ndvi_normalized.astype(np.uint8)
    ndvi_normalized = normal(ndvi_normalized,a,b)
    return ndvi_normalized
def histogram(image, file_name):
    # Tính histogram của ảnh
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Hiển thị histogram
    plt.figure(figsize=(10, 5))
    plt.hist(image.flatten(), bins=256, range=[20, 230], color='gray')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of the Image')
def crop_image(file_name):
    file = open(file_name[:-3]+'txt', 'r')
    line = file.readlines()
    file.close()
    numbers = line[0].split()
    numbers = [float(x) for x in numbers]

    image = cv2.imread(file_name)
    x = numbers[1] * image.shape[1]
    y = numbers[2] * image.shape[0]
    width = numbers[3] * image.shape[1]
    height = numbers[4] * image.shape[0]

    cropped_image = image[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
    name = file_name.split("/")
    # print(name[-1])
    resized_image = cv2.resize(cropped_image, (640,640))
    cv2.imwrite('crop/'+name[-1], resized_image)
    return Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
def segment(image, file_path):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    NVDI = RGN2NVDI(image)
    NVDI_gr = RGN2NVDI_green(image)

    _, segmented_image = cv2.threshold(NVDI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image = np.logical_not(segmented_image).astype(int)
    segmented_image = segmented_image.astype(np.uint8)

    _, segmented_image_gr = cv2.threshold(NVDI_gr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image_gr = np.logical_not(segmented_image_gr).astype(int)
    segmented_image_gr = segmented_image_gr.astype(np.uint8)

    mask = cv2.bitwise_and(segmented_image_gr,segmented_image_gr,mask=segmented_image)
    result = cv2.bitwise_and(image,image,mask=mask)
    # print(file_name)
    file_name = file_path.split("/")
    cv2.imwrite('/segment/' + file_name[-1], result)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
def RGN2NVDI_segment(image, file_path):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    file_name = file_path.split("/")
    if(file_name[-1][0] == 'r'): 
        b = 15
        a = 1/150
    elif (file_name[-1][0] == 'y'):
        b = 40
        a = 1/500
    else:
        a = 1/200
        b = 40
    red_band = image[:, :, 0].astype(float)
    nir_band = image[:, :, 2].astype(float)
    ndvi = (nir_band - red_band - 0.0001) / (nir_band + red_band + 0.0001)
    ndvi_normalized = (ndvi + 1) * 127.5
    ndvi_normalized = ndvi_normalized.astype(np.uint8)
    ndvi_normalized = normal(ndvi_normalized,a,b)
    # file_name = file_path.split("/")

    color_map = cv2.COLORMAP_JET  # Chọn bản đồ màu từ xanh đến đỏ
    color_mapped_image = cv2.applyColorMap(ndvi_normalized, color_map)  # Áp dụng bản đồ màu lên ảnh
    cv2.imwrite(current_folder_path + '/NDVI/' + file_name[-1],color_mapped_image)
    return ndvi_normalized
def encode_image_to_string(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string
def decode_image(image_string):
    decoded_image = base64.b64decode(image_string)
    nparr = np.frombuffer(decoded_image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

data = {'cay1':{
    "01_01_2000":{
        'name': 'cay1',
        'date': '01_01_2000',
        'values': '',
        'class': '1'
    }
}}

def tab1():
        # Function to handle Button 2 click
    global data
    global tree_chose
    def button1_click():
        global data
        def get_data_from_firebase():
            ref = db.reference('')
            return ref.get()
        # label1.config(text="Button 3 Clicked!")
        
        data = get_data_from_firebase()

        trees = list(data.keys())
        listbox1.delete(0, tk.END)
        for tree in trees:
            print(data[tree].keys())
            listbox1.insert(tk.END, "Cây " + str(tree[3:]))
        # print(data1)
        # trees = list(data.keys())

    # Create Button 1
    button1 = ttk.Button(tab1_frame, text="Cập nhật dữ liệu", command=button1_click)
    button1.pack(pady=10)
    button1.place(x=30, y=275, width=100, height=23)

    # Create a cây
    label2 = ttk.Label(tab1_frame, text="Cây")
    label2.pack(pady=10)
    label2.place(x=8, y=4, width=25, height=23)

    # Create a ngày
    label3 = ttk.Label(tab1_frame, text="Ngày")
    label3.pack(pady=10)
    label3.place(x=115, y=4, width=40, height=23)

    # Create a ngày
    label4 = ttk.Label(tab1_frame, text="Trạng thái:")
    label4.pack(pady=10)
    label4.place(x=300, y=275, width=70, height=23)


    # Create a Listbox
    listbox1 = tk.Listbox(tab1_frame)
    trees = list(data.keys())
    def on_listbox_select1(event):
        # Get the selected item
        global tree_chose
        if listbox1.curselection():
            selected_item = listbox1.get(listbox1.curselection())
            print("Selected item:", selected_item.split()[1])
            tree_chose = 'cay' + selected_item.split()[1]
            days = list(data['cay'+selected_item.split()[1]].keys())
            listbox2.delete(0,tk.END)
            for day in days:
                listbox2.insert(tk.END, str(day))
            

    for tree in trees:
        listbox1.insert(tk.END, "Cây " + str(tree[3:]))
    
    listbox1.bind("<<ListboxSelect>>", on_listbox_select1)
    listbox1.pack(pady=10)
    listbox1.place(x=8, y=30, width=95, height=238)

    def on_listbox_select2(event):
        # Get the selected item
        if listbox2.curselection():
            selected_item = listbox2.get(listbox2.curselection())
            if data[tree_chose][selected_item]['class'][1] == '0': status = "Bình thường"
            elif data[tree_chose][selected_item]['class'][1] == '1': status = "Thiếu Nito"
            else: status = "Thiếu nước"
            label = ttk.Label(tab1_frame, text=status)
            label.pack(pady=10)
            label.place(x=380, y=275, width=70, height=23)

            # print(data[tree_chose][selected_item]['values'])
            image = decode_image(data[tree_chose][selected_item]['values'])
            image = image.resize((280, 250))
            photo = ImageTk.PhotoImage(image)
            # label.place(x=17, y=17, width=300, height=300)
            label = tk.Label(tab1_frame, image=photo)
            label.place(x=230, y=15)
            # label.pack()
            tab1_frame.mainloop()


            


    # Create a ngày
    listbox2 = tk.Listbox(tab1_frame)
    listbox2.insert(tk.END, "10/11")
    listbox2.insert(tk.END, "09/11")
    listbox2.pack(pady=10)
    listbox2.place(x=115, y=30, width=95, height=238)
    listbox2.bind("<<ListboxSelect>>", on_listbox_select2)


# file_path = None
image = None
file_path = None
result = None
name = None

def tab2():
    # Function to handle Button 1 click
    def button1_click():
        # Mở hộp thoại chọn file
        global image
        global file_path
        global current_date 
        global name 
        name = 'cay1'
        current_date  = date.today()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        file_path = current_folder_path + '/raw_data' + file_path.split('raw_data')[1]
        
        label2 = ttk.Label(tab2_frame, text="Trạng thái")
        label2.pack(pady=10)
        label2.place(x=270, y=62, width=70, height=23)

        label6 = ttk.Label(tab2_frame, text='Ngày:  ')
        label6.pack(pady=10)
        label6.place(x=270, y=5, width=70, height=23)

        label6 = ttk.Label(tab2_frame, text=current_date)
        label6.pack(pady=10)
        label6.place(x=360, y=5, width=70, height=23)

        # Create a Label
        label4 = ttk.Label(tab2_frame, text="Cây")
        label4.pack(pady=10)
        label4.place(x=270, y=40, width=70, height=23)

        def on_combobox_select(event):
            global name
            name = 'cay'+combobox.get()
            print("Selected item:", name)
 
        combobox = ttk.Combobox(tab2_frame, values=["1", "2", "3", "4", "5", "6", "7", "8","9","10"])
        # Đặt giá trị mặc định cho Combobox
        combobox.set("Chọn tên cây")
        # Đặt vị trí và kích thước cho Combobox
        combobox.pack(pady=10)
        # Gắn sự kiện khi chọn mục trong Combobox
        combobox.place(x=370, y=40, width=70, height=23)
        combobox.bind("<<ComboboxSelected>>", on_combobox_select)
        # Kiểm tra xem người dùng đã chọn file hay chưa

        if file_path:
            image = crop_image(file_path)
            image = image.resize((250, 250))
            photo = ImageTk.PhotoImage(image)
            # label.place(x=17, y=17, width=300, height=300)
            label = tk.Label(tab2_frame, image=photo)
            label.place(x=5, y=5)
            # label.pack()
            tab2_frame.mainloop()

    # Function to handle Button 2 click
    def button2_click():
        global image
        global file_path
        global result
        # Create a Label
        label3 = ttk.Label(tab2_frame, text="")
        label3.pack(pady=10)
        label3.place(x=370, y=62, width=70, height=23)
        if image:
            image_seg = segment(image, file_path)
            NDVI_segment = RGN2NVDI_segment(image_seg, file_path)
            hist, bins = np.histogram(NDVI_segment.flatten(), bins=256, range=[20, 240])
            plt.plot(hist)
            # Save the plot as an image
            plt.savefig(file_path[:-3]+'png')
            # Open the saved image using PIL
            saved_image = Image.open(file_path[:-3]+'png')
            saved_image = saved_image.resize((200, 170)) 
            norm_hist = hist / sum(hist)
            loaded_model = pkl.load(open("kneighbor_model.pickle", "rb"))
            result = loaded_model.predict([norm_hist])
            if result == 0: status = "Bình thường"
            elif result == 1: status = "Thiếu Nito"
            else: status = "Thiếu nước"
            label3.config(text=status)

            # print(file_path.replace("raw_data/images", "NDVI"))
            NDVI_img = Image.open(file_path.replace("raw_data/images", "NDVI"))
            NDVI_img = NDVI_img.resize((200, 170)) 
            photo = ImageTk.PhotoImage(NDVI_img)
            label = tk.Label(tab2_frame, image=photo)
            label.place(x=265, y=85)
            tab2_frame.mainloop()

    # Function to handle Button 2 click
    def button3_click():
        global image
        global file_path
        global result
        # global current_date 
        global name
        # current_date_sps = current_date.split("/")
        # new_date = '11dddfff'
        current_date = str(date.today()).replace('-', '_')
        def store_data_to_firebase(name, current_date, values, class_image):
            ref = db.reference(name + '/' + current_date)
            # new_data = {
            #     current_date : {
            #     'name': name,
            #     'date': current_date,
            #     'values': values,
            #     'class': class_image
            # }
            # } 
            ref.set({
                'name': name,
                'date': current_date,
                'values': values,
                'class': class_image
            })
            print('Data written successfully.')

        # cred = credentials.Certificate('D:/FinalProject/tomato-6f924-firebase-adminsdk-5mh3b-1382727a0b.json')
        # firebase_admin.initialize_app(cred, {
        #     'databaseURL': 'https://tomato-6f924-default-rtdb.asia-southeast1.firebasedatabase.app/'
        # })  

        store_data_to_firebase(name, current_date, encode_image_to_string(file_path), str(result))
        label1 = ttk.Label(tab2_frame, text="Lưu trữ thành công!")
        label1.pack(pady=10)  
        label1.place(x=310, y=275, width=175, height=23)

    # Create Button 1
    button1 = ttk.Button(tab2_frame, text="Mở", command=button1_click)
    button1.pack(pady=10)
    button1.place(x=17, y=275, width=75, height=23)

    # Create Button 2
    button2 = ttk.Button(tab2_frame, text="Phân tích", command=button2_click)
    button2.pack(pady=10)
    button2.place(x=110, y=275, width=75, height=23)

    # Create Button 3
    button3 = ttk.Button(tab2_frame, text="Lưu trữ", command=button3_click)
    button3.pack(pady=10)
    button3.place(x=205, y=275, width=75, height=23)



# Create the main application window
root = tk.Tk()
root.title("Theo dõi cây trồng")
root.geometry("553x336")

# Create a Tab Control
tab_control = ttk.Notebook(root)

# Create the first tab
tab1_frame = ttk.Frame(tab_control)
tab_control.add(tab1_frame, text='Theo dõi')
tab_control.pack(expand=1, fill='both')
tab1()

# Create the second tab
tab2_frame = ttk.Frame(tab_control)
tab_control.add(tab2_frame, text='Phân tích')
tab2()

# Start the main event loop
root.mainloop()