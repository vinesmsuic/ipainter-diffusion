from repainter import RePainter, ImgPath2Tensor
import os
import argparse

def parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-s',"--source", default="out_inpaint",
                        help="filepath to input image folder. default=out_inpaint")
    parser.add_argument('-d',"--dest", default="out_inpaint_batch",
                        help="Destination folder for saving results. default=out_inpaint_batch")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="produce N different samples per image. default=5")
    return parser.parse_args()

def main(args):
    model = RePainter()
    helper = ImgPath2Tensor()
    # Batch Inference
    input_folder = args.source
    out_folder = args.dest
    num_samples = args.num_samples
    list_img = [os.path.join(input_folder,img) for img in os.listdir(input_folder) if (img.find("_")<0)]
    list_mask = [os.path.join(input_folder,img) for img in os.listdir(input_folder) if (img.find("_mask")>=0)]
    list_img.sort()
    list_mask.sort()
    for img_path, mask_path in zip(list_img, list_mask):
        print(img_path, mask_path)
        tensor_img = helper.get_tensor_from_img_path(img_path)
        tensor_mask = helper.get_tensor_from_img_path(mask_path, is_mask=True)
        model.infer_one_image_mult_sample(tensor_img, tensor_mask, img_path, n_samples=num_samples, target_folder=out_folder)

if __name__ == "__main__":
    args = parser()
    main(args)

