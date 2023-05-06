from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import requests
import json

# IDX2CLASS = json.load(open("./imagenet_class_index.json"))
IDX2CLASS = json.load(open("/Users/jongbeomkim/Desktop/workspace/explainable_ai/cam/imagenet_class_index.json"))

def load_image(url_or_path=""):
    url_or_path = str(url_or_path)

    if "http" in url_or_path:
        img_arr = np.asarray(
            bytearray(requests.get(url_or_path).content), dtype="uint8"
        )
        img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(url_or_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _convert_to_pil(copied_img)
    copied_img.show()


def _convert_to_array(img):
    img = np.array(img)
    return img


def _blend_two_images(img1, img2, alpha=0.5):
    img1 = _convert_to_pil(img1)
    img2 = _convert_to_pil(img2)
    blended_img = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _convert_to_array(blended_img)


def save_image(img, path):
    _convert_to_pil(img).save(str(path))


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def _reverse_jet_colormap(img):
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = list(map(tuple, _apply_jet_colormap(gray_values).reshape(256, 3)))
    color_to_gray_map = dict(zip(color_values, gray_values))

    out = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], axis=2, arr=img)
    return out


def draw_bboxes(img, bboxes):
    canvas = _convert_to_pil(img)
    draw = ImageDraw.Draw(canvas)

    for x1, y1, x2, y2, label in bboxes.values:
        draw.rectangle(
            xy=(x1, y1, x2, y2),
            outline=(255, 0, 0),
            fill=None,
            width=max(1, int(min(x2 - x1, y2 - y1) * 0.02))
        )

    for x1, y1, x2, y2, label in bboxes.values:
        draw.text(
            xy=(x1, y1 - 4),
            text=" " + IDX2CLASS[str(label)][1],
            fill="white",
            stroke_fill="black",
            stroke_width=2,
            font=ImageFont.truetype(
                # font="./fonts/Pretendard-Regular.otf",
                font="/Users/jongbeomkim/Downloads/Pretendard-1.3.6/public/static/Pretendard-Regular.otf",
                size=max(10, int(min(40, min(x2 - x1, y2 - y1) * 0.12)))
            ),
            anchor="la"
        )
    return canvas
