# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import logging
import os
import time
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from marigold import MarigoldPipeline
from src.util.autoSelectGPU import select_best_gpu

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
EXTENSION_LIST = [".jpg", ".jpeg", ".png", ".tif"]


def generate_heatmap(diff):
    diff = Normalize()(diff) * 255
    diff = diff.astype(np.uint8)

    cmap = plt.get_cmap('jet')
    heatmap = cmap(diff / 255.0)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    return heatmap


def generate_histogram_image(dem_array, temp_array, histogram_image_path, bins=128, dpi=800):

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    ax.hist(
        [dem_array.flatten(), temp_array.flatten()],
        bins=bins,
        range=(0, 255),
        label=['Target DEM', 'Predicted DEM'],
        color=['blue', 'red'],
        alpha=0.7,
        rwidth=0.9,
        align='mid',
    )

    ax.set_xlabel('Depth Value', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Histogram of Target and Predicted Depth Maps', fontsize=16)
    ax.legend(fontsize=12)

    ax.set_xlim(0, 255)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(histogram_image_path)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-lcm-v1-0",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        default="",
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. For the original (DDIM) version, it's recommended to use 10-50 steps, while for LCM 1-4 steps.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )
    parser.add_argument(
        "--test_dem_dir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--text_prompt_path",
        type=str,
        help="Path to the text prompt file.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    test_dem_dir = args.test_dem_dir
    output_dir = args.output_dir

    if input_rgb_dir == "" and test_dem_dir == "":
        input_rgb_dir = ["BASE_DATA_DIR/ImageToDEM-JK/singleRGBNormalizationTest/png-stretched-unique",
                         "BASE_DATA_DIR/ImageToDEM-东南亚/singleRGBNormalizationTest/png-stretched-unique",
                         "BASE_DATA_DIR/ImageToDEM-希腊/singleRGBNormalizationTest/png-stretched-unique"]
        test_dem_dir = [
            "/home/wrz/src/python/Marigold/BASE_DATA_DIR/ImageToDEM-JK/singleRGBNormalizationTest/DEM_255-unique",
            "/home/wrz/src/python/Marigold/BASE_DATA_DIR/ImageToDEM-东南亚/singleRGBNormalizationTest/DEM_255-unique",
            "/home/wrz/src/python/Marigold/BASE_DATA_DIR/ImageToDEM-希腊/singleRGBNormalizationTest/DEM_255-unique"]

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Output directories
    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_tif = os.path.join(output_dir, "depth_bw")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    output_dir_compare = os.path.join(output_dir, "compare")
    output_dir_histogram = os.path.join(output_dir, "histogram")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_compare, exist_ok=True)
    os.makedirs(output_dir_histogram, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = select_best_gpu()
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    if not isinstance(input_rgb_dir, list):
        rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    else:
        rgb_filename_list = []
        for dir in input_rgb_dir:
            rgb_filename_list.extend(glob(os.path.join(dir, "*")))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    text_prompt_path = args.text_prompt_path if hasattr(args, "text_prompt_path") else None
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)

    if test_dem_dir is not None:
        if not isinstance(test_dem_dir, list):

            dem_filename_list = glob(os.path.join(test_dem_dir, "*"))
            dem_filename_list = [
                f for f in dem_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
            ]
            dem_filename_list = sorted(dem_filename_list)
            n_dems = len(dem_filename_list)
            if n_dems > 0:
                logging.info(f"Found {n_dems} DEMs")
            else:
                logging.error(f"No DEM found in '{test_dem_dir}'")
                exit(1)
        else:
            dem_filename_list = []
            for dir in test_dem_dir:
                dem_filename_list.extend(glob(os.path.join(dir, "*")))
            dem_filename_list = [
                f for f in dem_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
            ]
            dem_filename_list = sorted(dem_filename_list)
            n_dems = len(dem_filename_list)
            if n_dems > 0:
                logging.info(f"Found {n_dems} DEMs")
            else:
                logging.error(f"No DEM found in '{test_dem_dir}'")
                exit(1)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    # kwargs = {"proxies": {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}}
    pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; "
        f"color_map = {color_map}."
    )
    bestL1Dict = None
    targetNameList = None
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)
        targetDEMList = []
        predDEMList = []
        nameList = []
        diffList = []

        for rgb_path in tqdm(rgb_filename_list, desc="Estimating depth", leave=True):
            fileBaseName = os.path.basename(rgb_path)
            if targetNameList is not None and fileBaseName.replace("png", "") not in targetNameList:
                continue
            # Read input image
            input_image = Image.open(rgb_path)
            if text_prompt_path is None:
                text_prompt = None
            elif targetNameList is None:
                if isinstance(test_dem_dir, list):
                    textPath = os.path.join("BASE_DATA_DIR",
                                            "TextPrompt-" + rgb_path.split('/')[1].split("-")[1] + "-processed",
                                            fileBaseName.replace(".png", ".txt"))
                    with open(textPath, "r") as f:
                        text_prompt = f.read()
                        print(f"{textPath}读取文本提示：{text_prompt}")
                else:
                    textPath = os.path.join(text_prompt_path, fileBaseName.replace(".png", ".txt"))
                    with open(textPath, "r") as f:
                        text_prompt = f.read()
                        print(f"{textPath}读取文本提示：{text_prompt}")
            else:
                text_prompt = open(
                    "output/train_marigold_东南亚Text/new text/" + fileBaseName.replace(".png", "") + ".txt").read()

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Save as npy
            if isinstance(input_rgb_dir, list):
                rgb_name_base = rgb_path.split("BASE_DATA_DIR/ImageToDEM-")[1].split("/singleRGBNormalization")[0] + \
                                os.path.splitext(fileBaseName)[0]
            else:
                rgb_name_base = os.path.splitext(fileBaseName)[0]
            # Predict depth
            pipe_out = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
                text_prompt=text_prompt,
                # save_intermediate_path=os.path.join(output_dir, "intermediate"),
                # img_name=rgb_name_base
            )

            depth_pred: np.ndarray = pipe_out.depth_np
            depth_colored: Image.Image = pipe_out.depth_colored


            pred_name_base = rgb_name_base + "_pred"
            # npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            # if os.path.exists(npy_save_path):
            #     logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            # np.save(npy_save_path, depth_pred)

            # Save as 16-bit uint png
            depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
            png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")

            # Colorize
            colored_save_path = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )
            if os.path.exists(colored_save_path):
                logging.warning(
                    f"Existing file: '{colored_save_path}' will be overwritten"
                )

            # Compare
            compare_save_path = os.path.join(
                output_dir_compare, f"{pred_name_base}_compare.png"
            )

            hist_image_save_path = os.path.join(
                output_dir_histogram, f"{pred_name_base}_histogram.png"
            )
            if os.path.exists(compare_save_path):
                logging.warning(
                    f"Existing file: '{compare_save_path}' will be overwritten"
                )
            # compareImage把原图、label、预测结果和彩色热力图拼接在一起
            if test_dem_dir is not None:
                compareImage = Image.new(
                    "RGB",
                    (input_image.width * 4, input_image.height),  # 增加一列用于存放直方图
                    (255, 255, 255),
                )
                compareImage.paste(input_image, (0, 0))
                if isinstance(test_dem_dir, list):
                    dem_path = os.path.join(os.path.join("BASE_DATA_DIR", "ImageToDEM-" + rgb_path.split(
                        '/')[1].split("-")[1], "singleRGBNormalizationTest/DEM_255-unique",
                                                         fileBaseName.replace(".png", ".tif")))
                else:
                    dem_path = os.path.join(test_dem_dir, fileBaseName.replace(".png", ".tif"))
                dem = Image.open(dem_path)
                demArray = np.array(dem)
                compareImage.paste(dem, (input_image.width, 0))
                depth_pred = Image.fromarray(depth_to_save)

                tempArray = (np.array(depth_pred) - np.min(np.array(depth_pred))) / (
                        np.max(np.array(depth_pred)) - np.min(np.array(depth_pred))) * 255
                diffArray = demArray - np.array(tempArray)
                compareImage.paste(Image.fromarray(tempArray), (input_image.width * 2, 0))
                compareImage.paste(Image.fromarray(generate_heatmap(abs(diffArray))),
                                   (input_image.width * 3, 0))
                nowL1 = np.mean(np.abs(diffArray))
                print(f"计算{fileBaseName} dem和pred的L1 差值：{nowL1}")
                # 和bestL1Dict进行比较
                if bestL1Dict is not None:
                    # 获取当前tile的名称
                    tile_name = fileBaseName.split(".")[0]
                    # 获取当前tile的L1差值
                    lastL1 = bestL1Dict[tile_name + "_pred.png"]
                    if nowL1 < lastL1:
                        print(f"当前tile的L1差值更小：{tile_name} {nowL1} < {lastL1}")
                        with open("tempDict.txt", "a") as f:
                            f.write(f"当前tile的L1差值更小：{tile_name} {nowL1} < {lastL1}\n")
                        bestL1Dict[tile_name + "_pred.png"] = nowL1
                    else:
                        print(f"当前tile的L1差值更大：{tile_name} {nowL1} > {lastL1}. 不更新")

                        with open("tempDict.txt", "a") as f:
                            f.write(f"当前tile的L1差值更大：{tile_name} {nowL1} > {lastL1}. 不更新\n")
                        continue

                # 生成直方图并将其粘贴到compareImage中
                generate_histogram_image(demArray, tempArray, hist_image_save_path)
                # compareImage.paste(hist_image, (input_image.width * 4, 0))
                # hist_image.savefig(hist_image_save_path)

                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")
                compareImage.save(compare_save_path)

                # 加入平均值和标准差
                targetDEMList.append([np.mean(demArray), np.std(demArray)])
                predDEMList.append([np.mean(np.array(tempArray)), np.std(np.array(tempArray))])
                nameList.append(rgb_path)

                # 加入差值的最大值、最小值、平均值和标准差
                diffList.append([np.max(diffArray), np.min(diffArray), np.mean(diffArray), np.std(diffArray)])

            depth_colored.save(colored_save_path)
        # 写入日志
        if bestL1Dict is not None:
            print(f"优化后平均值：{np.mean(list(bestL1Dict.values()))}")
            print(f"bestL1Dict: {bestL1Dict}")
            with open("tempDict.txt", "a") as f:
                f.write(str(bestL1Dict))
        logCompareFile = os.path.join(output_dir, "compare.log")
        ff = open(logCompareFile, "w")
        # 写入当前时间
        ff.write("当前时间：" + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + "\n")
        for i in range(len(nameList)):
            print(
                f"{nameList[i].replace('BASE_DATA_DIR/ImageToDEM-JK/singleRGBNormalizationTest/png-stretched-unique/', ''):<10}: "
                f"target mean={targetDEMList[i][0]:<10.5f}, target std={targetDEMList[i][1]:<10.5f}, "
                f"pred mean={predDEMList[i][0]:<10.5f}, pred std={predDEMList[i][1]:<10.5f}, "
                f"max diff={diffList[i][0]:<10.5f}, min diff={diffList[i][1]:<10.5f}, "
                f"mean diff={diffList[i][2]:<10.5f}, std diff={diffList[i][3]:<10.5f}"
            )
            ff.write(
                f"{nameList[i].replace('BASE_DATA_DIR/ImageToDEM-JK/singleRGBNormalizationTest/png-stretched-unique/', ''):<10}: "
                f"target mean={targetDEMList[i][0]:<10.5f}, target std={targetDEMList[i][1]:<10.5f}, "
                f"pred mean={predDEMList[i][0]:<10.5f}, pred std={predDEMList[i][1]:<10.5f}, "
                f"max diff={diffList[i][0]:<10.5f}, min diff={diffList[i][1]:<10.5f}, "
                f"mean diff={diffList[i][2]:<10.5f}, std diff={diffList[i][3]:<10.5f}\n"
            )
        print(nameList)
        print(targetDEMList)
        print(predDEMList)
        print(diffList)
        ff.write("\n")
        ff.write("nameList: " + str(nameList) + "\n")
        ff.write("targetDEMList: " + str(targetDEMList) + "\n")
        ff.write("predDEMList: " + str(predDEMList) + "\n")
        ff.write("diffList: " + str(diffList) + "\n")
        ff.close()


def calculateAllHistogram():
    targetAreaName = "东南亚"
    labelDEMDir = f"BASE_DATA_DIR/ImageToDEM-{targetAreaName}/singleRGBNormalizationTest/DEM_255-unique"
    outputDEMDir = f"/nfs5/wrz16/src/python/Marigold/output/train_marigold_{targetAreaName}Text/depth_bw"

    if not isinstance(labelDEMDir, list):

        dem_filename_list = glob(os.path.join(labelDEMDir, "*"))
        dem_filename_list = [
            f for f in dem_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
        ]
        dem_filename_list = sorted(dem_filename_list)
        n_dems = len(dem_filename_list)
        if n_dems > 0:
            logging.info(f"Found {n_dems} DEMs")
        else:
            logging.error(f"No DEM found in '{labelDEMDir}'")
            exit(1)
    else:
        dem_filename_list = []
        for dir in labelDEMDir:
            dem_filename_list.extend(glob(os.path.join(dir, "*")))
        dem_filename_list = [
            f for f in dem_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
        ]
        dem_filename_list = sorted(dem_filename_list)
        n_dems = len(dem_filename_list)
        if n_dems > 0:
            logging.info(f"Found {n_dems} DEMs")
        else:
            logging.error(f"No DEM found in '{labelDEMDir}'")
            exit(1)
    predArray, labelArray = [], []
    diffDict = {}
    for i in tqdm(range(len(dem_filename_list))):
        labelDEMPath = dem_filename_list[i]
        labelDEM = Image.open(labelDEMPath)
        labelArray.append(np.array(labelDEM))
        predDEMPath = os.path.join(outputDEMDir, os.path.basename(labelDEMPath).replace(".tif", "_pred.png"))
        predDEM = Image.open(predDEMPath)
        temp = np.array(predDEM)
        temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp)) * 255
        predArray.append(temp)
        diffDict[os.path.basename(labelDEMPath)] = np.array(labelDEM).mean() - np.array(temp).mean()

    # 排序
    diffDict = dict(sorted(diffDict.items(), key=lambda item: item[1]))
    print(diffDict)
    generate_total_image_histogram_image(diffDict)
    # generate_histogram_image(np.array(labelArray), np.array(predArray),os.path.dirname(outputDEMDir) + "/allHistogram.png")


def generate_total_image_histogram_image(diffDict):
    # 设置高分辨率图像大小和dpi
    fig, ax = plt.subplots(figsize=(12, 8), dpi=800)

    # 绘制直方图，使用 `plt.hist()` 支持直接绘制多个数据集，横坐标是1，2，3...；纵坐标是diffDict中对应的差值
    ax.hist(
        [list(diffDict.values())],
        bins=256,
        range=(0, len(diffDict)),
        label=['Diff DEM'],
        color=['blue'],
        alpha=0.7,
        rwidth=0.9,  # 调整宽度
        align='mid',  # 对齐方式
    )

    # 设置标签和标题
    ax.set_xlabel('Depth Value', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Histogram of Target and Predicted Depth Maps', fontsize=16)
    ax.legend(fontsize=12)

    # 设置坐标轴范围
    ax.set_xlim(-255, 255)
    ax.grid(axis='y', linestyle='--', alpha=0.7)  # 增加网格线

    # 保存图像
    plt.tight_layout()
    plt.savefig("totalHistogram.png")
    plt.close(fig)


if "__main__" == __name__:
    main()
    # calculateAllHistogram()

# /home/wrz/miniconda3/envs/sd/bin/python /home/wrz/src/python/Marigold/run.py --input_rgb_dir '' --output_dir /nfs5/wrz16/src/python/Marigold/output/train_marigold_混合3Text85000 --checkpoint BASE_CKPT_DIR/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79 --denoise_steps 50 --processing_res 0 --test_dem_dir '' --text_prompt_path ''

# 瑞士：/home/wrz/miniconda3/envs/sd/bin/python /home/wrz/src/python/Marigold/run.py --input_rgb_dir BASE_DATA_DIR/ImageToDEM-瑞士2_512/singleRGBNormalizationTest/png-stretched-unique --output_dir /nfs5/wrz/src/python/Marigold/output/train_marigold_瑞士2_512 --checkpoint BASE_CKPT_DIR/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79 --denoise_steps 50 --processing_res 0 --test_dem_dir /home/wrz/src/python/Marigold/BASE_DATA_DIR/ImageToDEM-瑞士2_512/singleRGBNormalizationTest/DEM_255-unique

# 瑞士Text：/home/wrz/miniconda3/envs/sd/bin/python /home/wrz/src/python/Marigold/run.py --input_rgb_dir BASE_DATA_DIR/ImageToDEM-瑞士2_512/singleRGBNormalizationTest/png-stretched-unique --output_dir /nfs5/wrz/src/python/Marigold/output/train_marigold_瑞士2_512Text30000 --checkpoint BASE_CKPT_DIR/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79 --denoise_steps 50 --processing_res 0 --test_dem_dir /home/wrz/src/python/Marigold/BASE_DATA_DIR/ImageToDEM-瑞士2_512/singleRGBNormalizationTest/DEM_255-unique --text_prompt_path BASE_DATA_DIR/TextPrompt-processed-瑞士2_512

# 澳大利亚Text：/home/wrz/miniconda3/envs/sd/bin/python /home/wrz/src/python/Marigold/run.py --input_rgb_dir BASE_DATA_DIR/ImageToDEM-澳大利亚/singleRGBNormalizationTest/png-stretched-unique --output_dir /nfs5/wrz/src/python/Marigold/output/train_marigold_澳大利亚Text15000 --checkpoint BASE_CKPT_DIR/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79 --denoise_steps 50 --processing_res 0 --test_dem_dir /home/wrz/src/python/Marigold/BASE_DATA_DIR/ImageToDEM-澳大利亚/singleRGBNormalizationTest/DEM_255-unique --text_prompt_path BASE_DATA_DIR/TextPrompt-processed-澳大利亚

# JKRandomText：/home/wrz/miniconda3/envs/sd/bin/python /home/wrz/src/python/Marigold/run.py --input_rgb_dir BASE_DATA_DIR/ImageToDEM-JK/singleRGBNormalizationTest/png-stretched-unique --output_dir /nfs5/wrz/src/python/Marigold/output/train_marigold_JK --checkpoint BASE_CKPT_DIR/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79 --denoise_steps 50 --processing_res 0 --test_dem_dir /home/wrz/src/python/Marigold/BASE_DATA_DIR/ImageToDEM-JK/singleRGBNormalizationTest/DEM_255-unique --text_prompt_path BASE_DATA_DIR/TextPrompt-JK-processed-RandomLetter

# /home/wrz/miniconda3/envs/sd/bin/python /home/wrz/src/python/Marigold/run.py --input_rgb_dir BASE_DATA_DIR/ImageToDEM-东南亚/singleRGBNormalizationTest/png-stretched-unique --output_dir /nfs5/wrz/src/python/Marigold/output/train_marigold_东南亚RandomText20000 --checkpoint BASE_CKPT_DIR/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79 --denoise_steps 50 --processing_res 0 --test_dem_dir /home/wrz/src/python/Marigold/BASE_DATA_DIR/ImageToDEM-东南亚/singleRGBNormalizationTest/DEM_255-unique --text_prompt_path BASE_DATA_DIR/TextPrompt-东南亚-processed-RandomLetter

