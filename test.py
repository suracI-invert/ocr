from PIL import Image, ImageOps
import os

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

if __name__ == "__main__":
    list_fns = os.listdir('./data/new_public_test')[:10]
    img_sizes = []
    for fname in list_fns:
        img = Image.open(os.path.join('./data/new_public_test', fname))
        img.save(f'./ori_img/{fname}')
        img = resize_with_padding(img, (70, 32))
        img_sizes.append(img.size)
        img.save(f"./resize_img/{fname}")
    for i in range(len(img_sizes) - 1):
        if img_sizes[i][0] != img_sizes[i + 1][0] and img_sizes[i][1] != img_sizes[i + 1][1]:
            print(i)

    idx = 1
    oriImg = Image.open(os.path.join('./data/new_public_test', list_fns[idx]))
    resize = Image.open(os.path.join('./resize_img', list_fns[idx]))