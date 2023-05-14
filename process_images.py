import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import json

IDX2CLASS = json.load(open("/Users/jongbeomkim/Desktop/workspace/explainable_ai/cam/imagenet_class_index.json"))


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def _to_array(img):
    img = np.array(img)
    return img


def _to_3d(img):
    if img.ndim == 2:
        return np.dstack([img, img, img])
    else:
        return img


def show_image(img):
    copied = img.copy()
    copied = _to_pil(copied)
    copied.show()


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def _preprocess_image(img):
    if img.dtype == "bool":
        img = img.astype("uint8") * 255
        
    if img.ndim == 2:
        if (
            np.array_equal(np.unique(img), np.array([0, 255])) or
            np.array_equal(np.unique(img), np.array([0])) or
            np.array_equal(np.unique(img), np.array([255]))
        ):
            img = _to_3d(img)
        else:
            img = _apply_jet_colormap(img)
    return img


def _blend_two_images(img1, img2, alpha=0.5):
    img1 = _to_pil(img1)
    img2 = _to_pil(img2)
    img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _to_array(img_blended)


def save_image(img1, img2=None, alpha=0.5, path="") -> None:
    copied1 = _preprocess_image(
        _to_array(img1.copy())
    )
    if img2 is None:
        img_arr = copied1
    else:
        copied2 = _to_array(
            _preprocess_image(
                _to_array(img2.copy())
            )
        )
        img_arr = _to_array(
            _blend_two_images(img1=copied1, img2=copied2, alpha=alpha)
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if img_arr.ndim == 3:
        cv2.imwrite(
            filename=str(path), img=img_arr[:, :, :: -1], params=[cv2.IMWRITE_JPEG_QUALITY, 100]
        )
    elif img_arr.ndim == 2:
        cv2.imwrite(
            filename=str(path), img=img_arr, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
        )


def resize_image(img, w, h):
    resized_img = cv2.resize(src=img, dsize=(w, h))
    return resized_img


def get_width_and_height(img):
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    return w, h


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def _rgba_to_rgb(img):
    copied = img.copy().astype("float")
    copied[..., 0] *= copied[..., 3] / 255
    copied[..., 1] *= copied[..., 3] / 255
    copied[..., 2] *= copied[..., 3] / 255
    copied = copied.astype("uint8")
    copied = copied[..., : 3]
    return copied


def _reverse_jet_colormap(img):
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = list(map(tuple, _apply_jet_colormap(gray_values).reshape(256, 3)))
    color_to_gray_map = dict(zip(color_values, gray_values))

    out = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], axis=2, arr=img)
    return out


def draw_bboxes(img, bboxes):
    canvas = _to_pil(img)
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


def denormalize_array(img):
    copied = img.copy()
    copied -= copied.min()
    copied /= copied.max()
    copied *= 255.0
    copied = np.clip(a=copied, a_min=0, a_max=255).astype("uint8")
    return copied
