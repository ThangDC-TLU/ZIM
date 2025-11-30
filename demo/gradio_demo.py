import os, sys
# Cần đảm bảo rằng các module như zim_anything đã được cài đặt hoặc có trong sys.path
# Vì bạn đã có sys.path.append(os.getcwd()), giả sử các module nằm trong thư mục làm việc.
sys.path.append(os.getcwd())

import torch
import gradio as gr
# Cần đảm bảo gradio_image_prompter đã được cài đặt.
# !pip install gradio_image_prompter # chạy lệnh này nếu chưa cài
from gradio_image_prompter import ImagePrompter
import numpy as np
import cv2
from zim_anything import zim_model_registry, ZimPredictor, ZimAutomaticMaskGenerator
from zim_anything.utils import show_mat_anns

# --- Bắt đầu phần thay đổi cho Kaggle ---
# Cấu hình đường dẫn cho Kaggle
KAGGLE_INPUT_DIR = "/kaggle/input"
DATASET_NAME = "zim-vit-l-2092"
CKPT_PATH = os.path.join(KAGGLE_INPUT_DIR, DATASET_NAME)

def get_examples():
    # Kiểm tra đường dẫn ví dụ. Thay đổi nếu thư mục ví dụ là một phần của input dataset.
    # Giả sử thư mục 'examples' nằm trong thư mục làm việc hiện tại (os.getcwd())
    assets_dir = os.path.join(os.getcwd(), 'examples') 
    
    # Nếu thư mục 'examples' là một phần của dataset input:
    # assets_dir = os.path.join(CKPT_PATH, 'examples') 
    
    if not os.path.exists(assets_dir):
        print(f"Thư mục ví dụ không tìm thấy tại: {assets_dir}. Vui lòng tạo hoặc tải lên thư mục 'examples'.")
        return []

    images = os.listdir(assets_dir)
    # Lọc chỉ lấy file ảnh (cần thiết)
    image_files = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return [os.path.join(assets_dir, img) for img in image_files]
# --- Kết thúc phần thay đổi cho Kaggle ---


# Giữ nguyên các hàm khác
def get_shortest_axis(image):
    h, w, _ = image.shape
    return h if h < w else w

# ... [giữ nguyên các hàm reset_image, reset_example_image, run_amg, run_model, 
# reset_scribble, update_scribble, draw_point, draw_images, get_point_or_box_prompts] ...

# Thêm lại các hàm bị cắt ở trên để có thể copy-paste toàn bộ, hoặc đảm bảo các hàm này
# có trong file script bạn đang chạy. (Không thêm lại ở đây để tránh lặp lại mã)
def reset_image(image, prompts):
    if image is None:
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    else:
        image = image['image']
    predictor.set_image(image)
    prompts = dict()
    black = np.zeros(image.shape[:2], dtype=np.uint8)
    
    return (image, image, image, black, black, prompts)

def reset_example_image(image, prompts):
    if image is None:
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)

    predictor.set_image(image)
    prompts = dict()
    black = np.zeros(image.shape[:2], dtype=np.uint8)

    image_dict = {}
    image_dict['image'] = image
    image_dict['prompts'] = prompts

    return (image, image_dict, image, image, black, black, prompts)

def run_amg(image):
    masks = mask_generator.generate(image)
    masks_vis = show_mat_anns(image, masks)
    
    return masks_vis

def run_model(image, prompts):
    if not prompts:
        raise gr.Error(f'Please input any point or BBox')
    point_coords = None
    point_labels = None
    boxes = None
    zim_mask = None

    if "point" in prompts:
        point_coords, point_labels = [], []

        for type, pts in prompts["point"]:
            point_coords.append(pts)
            point_labels.append(type)
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

    if "bbox" in prompts:
        boxes = prompts['bbox']
        boxes = np.array(boxes)

    if "scribble" in prompts:
        point_coords, point_labels = [], []

        for pts in prompts["scribble"]:
            point_coords.append(np.flip(pts))
            point_labels.append(1)
        if len(point_coords) == 0:
            raise gr.Error("Please input any scribbles.")
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

    zim_mask, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=boxes,
        multimask_output=False,
    )
    zim_mask = np.squeeze(zim_mask, axis=0)
    zim_mask = np.uint8(zim_mask * 255)

    return zim_mask

def reset_scribble(image, scribble, prompts):
    # scribble = dict()
    for k in prompts.keys():
        prompts[k] = []

    for k, v in scribble.items():
        scribble[k] = None

    zim_mask = np.zeros_like(image)

    return scribble, zim_mask

def update_scribble(image, scribble, prompts):
    if "point" in prompts:
        del prompts["point"]

    if "bbox" in prompts:
        del prompts["bbox"]
    
    prompts = dict() # reset prompt
    scribble_mask = scribble["layers"][0][..., -1] > 0

    scribble_coords = np.argwhere(scribble_mask)
    n_points = min(len(scribble_coords), 24)
    indices = np.linspace(0, len(scribble_coords)-1, n_points, dtype=int)
    scribble_sampled = scribble_coords[indices]

    prompts["scribble"] = scribble_sampled
    
    zim_mask = run_model(image, prompts)

    return zim_mask, prompts


def draw_point(img, pt, size, color):
    # draw circle with white boundary region
    cv2.circle(img, (int(pt[0]), int(pt[1])), int(size * 1.3), (255, 255, 255), -1)
    cv2.circle(img, (int(pt[0]), int(pt[1])), int(size * 0.9), color, -1)


def draw_images(image, mask, prompts):
    if len(prompts) == 0 or mask.shape[1] == 1:
        return image, image, image

    minor = get_shortest_axis(image)
    size = int(minor / 80)

    image = np.float32(image)

    def blending(image, mask):
        mask = np.float32(mask) / 255
        blended_image = np.zeros_like(image, dtype=np.float32)
        blended_image[:, :, :] = [108, 0, 192]
        blended_image = (image * 0.5) + (blended_image * 0.5)
    
        img_with_mask = mask[:, :, None] * blended_image + (1 - mask[:, :, None]) * image
        img_with_mask = np.uint8(img_with_mask)

        return img_with_mask

    img_with_mask = blending(image, mask)
    img_with_point = img_with_mask.copy()

    if "point" in prompts:
        for type, pts in prompts["point"]:
            if type == 1: # type == "Positive": # Thay đổi từ string sang int (1)
                color = (0, 0, 255)
                draw_point(img_with_point, pts, size, color)
            elif type == 0: # type == "Negative": # Thay đổi từ string sang int (0)
                color = (255, 0, 0)
                draw_point(img_with_point, pts, size, color)

    # Đảm bảo trả về đủ 3 giá trị (img, img_with_mask, img_with_point)
    return image, img_with_mask, img_with_point


def get_point_or_box_prompts(img, prompts):
    image, img_prompts = img['image'], img['points']
    point_prompts = []
    box_prompts = []
    for prompt in img_prompts:
        for p in range(len(prompt)):
            prompt[p] = int(prompt[p])
        if prompt[2] == 2 and prompt[5] == 3:  # box prompt
            box_prompts = [[prompt[0], prompt[1], prompt[3], prompt[4]], ]
        elif prompt[2] == 1 and prompt[5] == 4:  # Positive point prompt
            point_prompts.append((1, (prompt[0], prompt[1])))
        elif prompt[2] == 0 and prompt[5] == 4:  # Negative point prompt
            point_prompts.append((0, (prompt[0], prompt[1])))

    if "scribble" in prompts:
        del prompts["scribble"]

    if len(point_prompts) > 0:
        # Sử dụng 1 cho Positive và 0 cho Negative như trong hàm run_model
        prompts['point'] = point_prompts 
    elif 'point' in prompts:
        del prompts['point']

    if len(box_prompts) > 0:
        prompts['bbox'] = box_prompts
    elif 'bbox' in prompts:
        del prompts['bbox']

    zim_mask = run_model(image, prompts)

    return image, zim_mask, prompts


if __name__ == "__main__":

    backbone = "vit_l"
    # --- SỬA ĐỔI CHÍNH Ở ĐÂY ---
    # Sử dụng đường dẫn checkpoint đã được xác định ở đầu
    ckpt_p = CKPT_PATH 
    # --- KẾT THÚC SỬA ĐỔI CHÍNH ---

    model = zim_model_registry[backbone](checkpoint=ckpt_p)
    if torch.cuda.is_available():
        model.cuda()
    
    predictor = ZimPredictor(model)
    mask_generator = ZimAutomaticMaskGenerator(
        model, 
        pred_iou_thresh=0.7, 
        points_per_batch=8,
        stability_score_thresh=0.9, 
    )
    
    with gr.Blocks() as demo:
        gr.Markdown("# <center> [Demo] ZIM: Zero-Shot Image Matting for Anything")

        prompts = gr.State(dict())
        img = gr.Image(visible=False)
        example_image = gr.Image(visible=False)
        
        with gr.Row():
            with gr.Column():
                # Point and Bbox prompt
                with gr.Tab(label="Point or Box"):
                    img_with_point_or_box = ImagePrompter(
                        label="query image", 
                        sources="upload"
                    )
                    interactions = "Left Click (Pos) | Middle/Right Click (Neg) | Press Move (Box)"
                    gr.Markdown("<h3 style='text-align: center'>[🖱️] 🌟 {} 🌟 </h3>".format(interactions))
                    run_bttn = gr.Button("Run")
                    amg_bttn = gr.Button("Automatic Mask Generation")
                    
                # Scribble prompt
                with gr.Tab(label="Scribble"):
                    img_with_scribble = gr.ImageEditor(
                        label="Scribble", 
                        brush=gr.Brush(colors=["#00FF00"], default_size=15),
                        sources="upload", 
                        transforms=None, 
                        layers=False
                    )
                    interactions = "Press Move (Scribble)"
                    gr.Markdown("<h3 style='text-align: center'> Step 1. Select Draw button </h3>")
                    gr.Markdown("<h3 style='text-align: center'> Step 2. 🌟 {} 🌟 </h3>".format(interactions))
                    scribble_bttn = gr.Button("Run")
                    scribble_reset_bttn = gr.Button("Reset Scribbles")
                    amg_scribble_bttn = gr.Button("Automatic Mask Generation")
                
                # Example image
                gr.Examples(get_examples(), inputs=[example_image])

            # with gr.Row():
            with gr.Column():
                # Thêm một tab cho ảnh có điểm/box được vẽ lên
                with gr.Tab(label="ZIM Image with Prompts"):
                    img_with_point_vis = gr.Image(
                        label="ZIM Image with Prompts", 
                        interactive=False
                    )

                with gr.Tab(label="ZIM Image"):
                    img_with_zim_mask = gr.Image(
                        label="ZIM Image", 
                        interactive=False
                    )

                with gr.Tab(label="ZIM Mask"):
                    zim_mask = gr.Image(
                        label="ZIM Mask", 
                        image_mode="L", 
                        interactive=False
                    )
                with gr.Tab(label="ZIM Auto Mask"):
                    zim_amg = gr.Image(
                        label="ZIM Auto Mask", 
                        interactive=False
                    )
                    
        example_image.change(
            reset_example_image,
            [example_image, prompts],
            [
                img,
                img_with_point_or_box,
                img_with_scribble,
                img_with_zim_mask,
                zim_amg,
                zim_mask,
                prompts,
            ]
        )

        img_with_point_or_box.upload(
            reset_image,
            [img_with_point_or_box, prompts],
            [
                img,
                img_with_scribble,
                img_with_zim_mask,
                zim_amg,
                zim_mask,
                prompts,
            ],
        )

        amg_bttn.click(
            run_amg,
            [img],
            [zim_amg]
        )
        amg_scribble_bttn.click(
            run_amg,
            [img],
            [zim_amg]
        )
        
        run_bttn.click(
            get_point_or_box_prompts,
            [img_with_point_or_box, prompts],
            [img, zim_mask, prompts]
        )

        zim_mask.change(
            draw_images,
            [img, zim_mask, prompts],
            [
                img, img_with_zim_mask, img_with_point_vis # Cập nhật đầu ra của draw_images
            ],
        )
        scribble_reset_bttn.click(
            reset_scribble,
            [img, img_with_scribble, prompts],
            [img_with_scribble, zim_mask],
        )
        scribble_bttn.click(
            update_scribble,
            [img, img_with_scribble, prompts],
            [zim_mask, prompts],
        )

    demo.queue()
    # --- SỬA ĐỔI CHÍNH Ở ĐÂY ---
    # Chạy launch Gradio trong môi trường Notebook
    demo.launch(
        # Xóa server_name và server_port để Gradio tự động hiển thị
        # Có thể dùng debug=True nếu cần kiểm tra lỗi
    )
    # --- KẾT THÚC SỬA ĐỔI CHÍNH ---