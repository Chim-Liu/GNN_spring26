import numpy as np
import struct
from PIL import Image

def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        # 读取文件头信息
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # 读取像素数据
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images

# 填入你解压后的文件路径，比如 train-images-idx3-ubyte
images = read_mnist_images('mnist/train-images-idx3-ubyte/train-images-idx3-ubyte')

# 保存前 5 张为普通图片，方便你上传
for i in range(5):
    img = Image.fromarray(images[i])
    img.save(f'mnist_sample_{i}.png')
    print(f'已生成第 {i} 张样本图')