import os
from PIL import Image
from torch.utils.data import Dataset
from compose_img import combine_image, remove_ds_store, traverse_folder
from image_caption import show_n_generate
from extractor import get_keyword
from ask_llm import ask_gpt
import argparse

path_driver = '/Users/jin666/Desktop/jmy_generate/gpt-test/chromedriver_mac_arm64/chromedriver'  # diver
username = "18201768019"
password = "chat8app"

parser = argparse.ArgumentParser()
parser.add_argument('--target_size', type=tuple, default='(256,256)', help='The input image size.')
parser.add_argument('--folder_path', type=str, default='/Users/jin666/Desktop/jmy_generate/input', help='The folder path.')
parser.add_argument('--path_driver', type=str, default='/Users/jin666/Desktop/jmy_generate/gpt-test/chromedriver_mac_arm64/chromedriver', help='11')
parser.add_argument('--username', type=str, default='18201768019', help='username.')
parser.add_argument('--password', type=str, default='chat8app', help='password.')
opt = parser.parse_args()

target_size = (256, 256)
folder_path = '/Users/jin666/Desktop/jmy_generate/input'

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_list = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.folder_path, self.image_list[index])
        image = Image.open(image_path)
        image = image.resize(target_size)

        if self.transform:
            image = self.transform(image)

        return image


transform = None
dataset = ImageFolderDataset(folder_path, transform=transform)

# 调用方法删除 .DS_Store 文件
remove_ds_store(folder_path)
# 取出语义
list1 = traverse_folder(folder_path)
list_sematic = []
for i in list1:
    text = show_n_generate(i)
    list_sematic.append(text)
print(list_sematic)
keyword_list = get_keyword(list_sematic)
# enter gpt and get sentence
o_answer = ask_gpt(path_driver,username,password,keyword_list)

# Open the nine images you want to combine
combine_image(target_size, dataset, o_answer)
print(o_answer)