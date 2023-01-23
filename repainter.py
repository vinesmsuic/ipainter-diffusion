#==================================================
# Author: vinesmsuic
#+=================================================

import os
import torch as th
import torch.nn.functional as F
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    select_args,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

class RePainter():
    def __init__(self) -> None:
        self.conf = conf_mgt.conf_base.Default_Conf()
        self.conf.update(yamlread("repaint_config.yml"))
        print("Start", self.conf['name'])
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        print("Device: ", self.device)
        self.model, self.diffusion = create_model_and_diffusion(
            **select_args(self.conf, model_and_diffusion_defaults().keys()), conf=self.conf
        )
        self.model.load_state_dict(
            dist_util.load_state_dict(os.path.join("weights", "places256_300000.pt"), map_location="cpu")
        )
        self.model.to(self.device)
        if self.conf.use_fp16:
            self.model.convert_to_fp16()
        self.model.eval()
        self.show_progress = self.conf.show_progress

    def infer_one_image(self, image, mask, file_path, progressive=False):
        def model_fn(x, t, y=None, gt=None, **kwargs):
            return self.model(x, t, y if self.conf.class_cond else None, gt=gt)


        print("sampling...")
        image = image.to(self.device)
        mask = mask.to(self.device)

        model_kwargs = {}

        model_kwargs["gt"] = image

        gt_keep_mask = mask
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if progressive:
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.axis('off')
            ims = []
            for sample in self.diffusion.p_sample_loop_progressive(
                model_fn,
                (batch_size, 3, self.conf.image_size, self.conf.image_size),
                clip_denoised=self.conf.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
                device=self.device,
                progress=self.show_progress,
                conf=self.conf
            ):
                img = toU8(sample["sample"]).squeeze(0)
                ims.append(img)
            
            last_percent = len(ims) //10
            ims = ims[last_percent:]
            i = 0
            out_dir = 'intermediate_results'
            os.makedirs(out_dir, exist_ok=True)
            for im in ims:
                outpath = 'last10percent_' + str(i)
                ext = '.png'
                Image.fromarray(im).save(os.path.join(out_dir, outpath+ext))
                i += 1

            """
            # Code for saving a animated GIf.
                img = toU8(sample["sample"]).squeeze(0)
                sample_img = plt.imshow(img, animated=True)
                ims.append([sample_img])


            Image.fromarray(img).save('inpainted.png')
            animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
            animate.save('inpainted.gif')
            """

        else:
            result = self.diffusion.p_sample_loop(
                model_fn,
                (batch_size, 3, self.conf.image_size, self.conf.image_size),
                clip_denoised=self.conf.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=self.device,
                progress=self.show_progress,
                return_all=True,
                conf=self.conf
            )

            srs = toU8(result['sample'])
            gts = toU8(result['gt'])
            lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                    th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

            gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

            #basename = os.path.basename(file_path)
            name, ext = os.path.splitext(file_path)
            #Image.fromarray(np.squeeze(gts, axis=0)).save(name + '_gt' + ext)
            #Image.fromarray(np.squeeze(lrs, axis=0)).save(name + '_lr' + ext)
            #Image.fromarray(np.squeeze(gt_keep_masks, axis=0)).save(name + '_gt_keep_masks' + ext)
            Image.fromarray(np.squeeze(srs, axis=0)).save(name + '_inpainted' + ext)

            print("sampling complete")
            return name + '_inpainted' + ext

    def infer_one_image_mult_sample(self, image, mask, file_path, n_samples=10, target_folder=None):
        def model_fn(x, t, y=None, gt=None, **kwargs):
            return self.model(x, t, y if self.conf.class_cond else None, gt=gt)

        print("sampling...")
        image = image.to(self.device)
        mask = mask.to(self.device)

        model_kwargs = {}

        model_kwargs["gt"] = image

        gt_keep_mask = mask
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        name, ext = os.path.splitext(file_path)
        basename_noext = os.path.splitext(os.path.basename(file_path))[0]

        for idx in range(n_samples):
            result = self.diffusion.p_sample_loop(
                model_fn,
                (batch_size, 3, self.conf.image_size, self.conf.image_size),
                clip_denoised=self.conf.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=self.device,
                progress=self.show_progress,
                return_all=False,
                conf=self.conf
            )

            srs = toU8(result)
            PIL_srs = Image.fromarray(np.squeeze(srs, axis=0))

            if target_folder is None:
                PIL_srs.save(name + '_inpainted_' + str(idx) + ext)
            else:
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                target_name = os.path.join(target_folder, basename_noext + '_inpainted_' + str(idx) + ext)
                PIL_srs.save(target_name)
            
        print("sampling complete")

class ImgPath2Tensor():
    def __init__(self, width=256, height=256) -> None:
        self.transform_inference = A.Compose(
            [   
                A.Resize(width=width, height=height),
                #A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ]
        )

    def get_tensor_from_pil(self, img_pil, is_mask:bool=False):
        img = img_pil.convert("RGB")
        img = np.array(img)
        img = self.transform_inference(image=img)["image"]
        tensor_img = img.unsqueeze(0)
        if is_mask:
            tensor_img = tensor_img / 255.0
        else:
            tensor_img = tensor_img / 127.5 - 1
        return tensor_img

    def get_tensor_from_img_path(self, img_path, is_mask:bool=False):
        # We normalize image and mask in a different way
        pil_image = Image.open(img_path).convert("RGB")
        tensor_img = self.get_tensor_from_pil(pil_image, is_mask=is_mask)
        return tensor_img

if __name__ == "__main__":
    model = RePainter()
    helper = ImgPath2Tensor()
    # Batch Inference
    input_folder = 'out_inpaint'
    out_folder = 'out_inpaint_batch'
    list_img = [os.path.join(input_folder,img) for img in os.listdir(input_folder) if (img.find("_")<0)]
    list_mask = [os.path.join(input_folder,img) for img in os.listdir(input_folder) if (img.find("_mask")>=0)]
    list_img.sort()
    list_mask.sort()
    for img_path, mask_path in zip(list_img, list_mask):
        print(img_path, mask_path)
        tensor_img = helper.get_tensor_from_img_path(img_path)
        tensor_mask = helper.get_tensor_from_img_path(mask_path, is_mask=True)
        model.infer_one_image_mult_sample(tensor_img, tensor_mask, img_path, n_samples=5, target_folder=out_folder) # n_samples = produce N different image
