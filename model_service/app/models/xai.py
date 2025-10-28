import logging
import torch
import numpy as np
from captum.attr import LayerGradCam
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

def unnormalize_tensor(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).reshape(1, 3, 1, 1).to(tensor.device)
    return (tensor * std) + mean

def run_grad_cam(model, normalized_tensor, unnormalized_tensor, target_label_idx, device):
    """
    Chạy Grad-CAM và trả về ảnh heatmap dưới dạng base64.
    """
    model.eval() 

    target_layer = model.layer4 

    try:
        lgc = LayerGradCam(model, target_layer)

        attributions = lgc.attribute(
            normalized_tensor.to(device),
            target=target_label_idx
        )

        img_np = unnormalized_tensor.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)  # ensure in [0,1]

        attr_np = attributions.cpu().detach().numpy().squeeze()

        # attr_np may be (H,W) or (H,W,3). Normalize to (H,W).
        if attr_np.ndim == 3:
            # average across channels
            attr_gray = np.mean(attr_np.transpose(1, 2, 0), axis=2)
        elif attr_np.ndim == 2:
            attr_gray = attr_np
        else:
            # unexpected shape -> fallback to zeros
            attr_gray = np.zeros((img_np.shape[0], img_np.shape[1]))

        # Try using captum visualization first; if it doesn't return a figure, fallback
        try:
            fig, _ = viz.visualize_image_attr(
                attr_gray,
                img_np,
                method="blended_heat_map",
                sign="all",
                show_colorbar=False,
                title="",
                use_pyplot=False
            )
            if fig is None:
                raise RuntimeError("captum.visualize_image_attr returned None fig")
        except Exception:
            fig = plt.figure(figsize=(4, 4), dpi=100)
            plt.imshow(img_np)
            a = attr_gray
            try:
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            except Exception:
                a = np.zeros_like(a)
            plt.imshow(a, cmap='jet', alpha=0.45)
            plt.axis('off')

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        logger.exception("Error running Grad-CAM: %s", e)
        return ""