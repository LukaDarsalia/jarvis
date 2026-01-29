from PIL import Image
import numpy as np
import cv2
import copy


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式 

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上
    
    
    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    
    # 对掩码进行高斯模糊，使边缘更平滑
    blur_kernel_size = int(0.05 * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊
    #mask_array = np.array(modified_mask_image)
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def _extract_crop_with_padding(image: np.ndarray, x_s: int, y_s: int, x_e: int, y_e: int) -> np.ndarray:
    """Extract crop (with zero padding) for out-of-bounds boxes."""
    h, w = image.shape[:2]
    crop_w = max(0, x_e - x_s)
    crop_h = max(0, y_e - y_s)
    if crop_w == 0 or crop_h == 0:
        return np.zeros((0, 0, 3), dtype=image.dtype)

    crop = np.zeros((crop_h, crop_w, 3), dtype=image.dtype)
    src_x0 = max(0, x_s)
    src_y0 = max(0, y_s)
    src_x1 = min(w, x_e)
    src_y1 = min(h, y_e)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return crop
    dst_x0 = src_x0 - x_s
    dst_y0 = src_y0 - y_s
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    crop[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return crop


def _paste_crop_with_padding(image: np.ndarray, crop: np.ndarray, x_s: int, y_s: int, x_e: int, y_e: int) -> None:
    """Paste crop back into image with bounds checking."""
    h, w = image.shape[:2]
    src_x0 = max(0, x_s)
    src_y0 = max(0, y_s)
    src_x1 = min(w, x_e)
    src_y1 = min(h, y_e)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return
    dst_x0 = src_x0 - x_s
    dst_y0 = src_y0 - y_s
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    image[src_y0:src_y1, src_x0:src_x1] = crop[dst_y0:dst_y1, dst_x0:dst_x1]


def get_image_blending(image, face, face_box, mask_array, crop_box):
    """Fast numpy-based blending to replace PIL-heavy path."""
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box

    # Extract crop with padding (to mimic PIL behavior for out-of-bounds boxes)
    face_large = _extract_crop_with_padding(image, x_s, y_s, x_e, y_e)
    if face_large.size == 0:
        return image

    # Normalize mask to single-channel float alpha
    if isinstance(mask_array, Image.Image):
        mask = np.array(mask_array.convert("L"))
    else:
        mask = np.array(mask_array)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

    crop_h, crop_w = face_large.shape[:2]
    if mask.shape[0] != crop_h or mask.shape[1] != crop_w:
        mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    # Place face patch into crop
    fx0 = x - x_s
    fy0 = y - y_s
    fx1 = fx0 + (x1 - x)
    fy1 = fy0 + (y1 - y)
    face_large_with_patch = face_large.copy()
    # Clamp to crop bounds just in case
    px0 = max(0, fx0)
    py0 = max(0, fy0)
    px1 = min(crop_w, fx1)
    py1 = min(crop_h, fy1)
    if px1 > px0 and py1 > py0:
        src_x0 = px0 - fx0
        src_y0 = py0 - fy0
        src_x1 = src_x0 + (px1 - px0)
        src_y1 = src_y0 + (py1 - py0)
        face_large_with_patch[py0:py1, px0:px1] = face[src_y0:src_y1, src_x0:src_x1]

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = (alpha * face_large_with_patch + (1.0 - alpha) * face_large).astype(np.uint8)

    out = image.copy()
    _paste_crop_with_padding(out, blended, x_s, y_s, x_e, y_e)
    return out


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
