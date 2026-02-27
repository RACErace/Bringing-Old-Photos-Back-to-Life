# 老照片修复（官方 PyTorch 实现）

<img src='imgs/0001.jpg'/>

### [项目主页](http://raywzy.com/Old_Photo/) | [论文（CVPR 版本）](https://arxiv.org/abs/2004.09484) | [论文（期刊版本）](https://arxiv.org/pdf/2009.07047v1.pdf) | [预训练模型](https://hkustconnect-my.sharepoint.com/:f:/g/personal/bzhangai_connect_ust_hk/Em0KnYOeSSxFtp4g_dhWdf0BdeT3tY12jIYJ6qvSf300cA?e=nXkJH2) | [Colab 演示](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing) | [Replicate 演示 & Docker 镜像](https://replicate.ai/zhangmozhe/bringing-old-photos-back-to-life) :fire:

**Bringing Old Photos Back to Life，CVPR2020（口头报告）**

**Old Photo Restoration via Deep Latent Space Translation，TPAMI 2022**

[Ziyu Wan（万子瑜）](http://raywzy.com/)<sup>1</sup>，
[Bo Zhang（张博）](https://www.microsoft.com/en-us/research/people/zhanbo/)<sup>2</sup>，
[Dongdong Chen（陈东东）](http://www.dongdongchen.bid/)<sup>3</sup>，
[Pan Zhang（张盼）](https://panzhang0212.github.io/)<sup>4</sup>，
[Dong Chen（陈栋）](https://www.microsoft.com/en-us/research/people/doch/)<sup>2</sup>，
[Jing Liao（廖菁）](https://liaojing.github.io/html/)<sup>1</sup>，
[Fang Wen（文芳）](https://www.microsoft.com/en-us/research/people/fangwen/)<sup>2</sup> <br>
<sup>1</sup>香港城市大学，<sup>2</sup>微软亚洲研究院，<sup>3</sup>微软云 AI，<sup>4</sup>中国科学技术大学

## :sparkles: 最新动态
**2022.3.31**：我们关于老电影修复的新工作将在 CVPR 2022 上发表。更多详情请参阅[项目网站](http://raywzy.com/Old_Film/)和 [GitHub 仓库](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life)。

本框架现已支持高分辨率输入的修复。

<img src='imgs/HR_result.png'>

训练代码已开放，欢迎试用并了解训练细节。

您现在可以使用我们的 [Colab](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing) 在线体验，用您自己的照片试试看。

## 环境要求
代码在安装了 Nvidia GPU 和 CUDA 的 Ubuntu 系统上测试通过。运行代码需要 Python>=3.6。

## 安装

克隆 Synchronized-BatchNorm-PyTorch 仓库：

```
cd Face_Enhancement/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../../
```

```
cd Global/detection_models
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

下载人脸关键点检测预训练模型：

```
cd Face_Detection/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd ../
```

下载预训练模型，将文件 `Face_Enhancement/checkpoints.zip` 放在 `./Face_Enhancement` 目录下，将文件 `Global/checkpoints.zip` 放在 `./Global` 目录下，然后分别解压。

```
cd Face_Enhancement/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip
unzip face_checkpoints.zip
cd ../
cd Global/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
unzip global_checkpoints.zip
cd ../
```

安装依赖：

```bash
pip install -r requirements.txt
```

## :rocket: 如何使用？

**注意**：GPU 可以设置为 0 或 0,1,2 或 0,2；使用 -1 表示 CPU 模式

### 1）完整流程

安装完成并下载预训练模型后，您可以通过一个简单的命令轻松修复老照片。

对于没有划痕的图片：

```
python run.py --input_folder [测试图片文件夹路径] \
              --output_folder [输出路径] \
              --GPU 0
```

对于有划痕的图片：

```
python run.py --input_folder [测试图片文件夹路径] \
              --output_folder [输出路径] \
              --GPU 0 \
              --with_scratch
```

**对于有划痕的高分辨率图片**：

```
python run.py --input_folder [测试图片文件夹路径] \
              --output_folder [输出路径] \
              --GPU 0 \
              --with_scratch \
              --HR
```

注意：请尽量使用绝对路径。最终结果将保存在 `./output_path/final_output/` 中。您也可以在 `output_path` 中查看不同步骤的中间结果。

### 2）划痕检测

目前我们暂不打算直接发布带标签的老照片划痕数据集。如果您需要配对数据，可以使用我们的预训练模型对采集的图片进行测试以获取标签。

```
cd Global/
python detection.py --test_path [测试图片文件夹路径] \
                    --output_dir [输出路径] \
                    --input_size [resize_256|full_size|scale_256]
```

<img src='imgs/scratch_detection.png'>

### 3）全局修复

我们提出了一个三元域转换网络来同时解决老照片的结构化退化和非结构化退化问题。

<p align="center">
<img src='imgs/pipeline.PNG' width="50%" height="50%"/>
</p>

```
cd Global/
python test.py --Scratch_and_Quality_restore \
               --test_input [测试图片文件夹路径] \
               --test_mask [对应的掩码] \
               --outputs_dir [输出路径]

python test.py --Quality_restore \
               --test_input [测试图片文件夹路径] \
               --outputs_dir [输出路径]
```

<img src='imgs/global.png'>

### 4）人脸增强

我们使用渐进式生成器来优化老照片中的人脸区域。更多细节请参阅我们的期刊论文和 `./Face_Enhancement` 文件夹。

<p align="center">
<img src='imgs/face_pipeline.jpg' width="60%" height="60%"/>
</p>

<img src='imgs/face.png'>

> *注意*：
> 本仓库主要用于研究目的，我们尚未对运行性能进行优化。
>
> 由于模型是使用 256×256 的图片进行预训练的，因此对于任意分辨率的图片可能无法达到理想效果。

### 5）图形用户界面（GUI）

一个用户友好的图形界面，用户可以输入图片并在相应窗口中查看结果。

#### 使用方法：

1. 运行 GUI.py 文件。
2. 点击"浏览"按钮，从 test_images/old_w_scratch 文件夹中选择您的图片以去除划痕。
3. 点击"修改照片"按钮。
4. 等待片刻，在 GUI 窗口中查看结果。
5. 点击"退出窗口"关闭界面，修复后的图片将保存在 output 文件夹中。

<img src='imgs/gui.PNG'>

## 如何训练？

### 1）创建训练文件

将 VOC 数据集文件夹、采集的老照片文件夹（例如 Real_L_old 和 Real_RGB_old）放入同一个共享文件夹中，然后执行：

```
cd Global/data/
python Create_Bigfile.py
```

注意：请根据您自己的环境修改代码。

### 2）分别训练域 A 和域 B 的 VAE

```
cd ..
python train_domain_A.py --use_v2_degradation --continue_train --training_dataset domain_A --name domainA_SR_old_photos --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [您的数据文件夹] --no_instance --resize_or_crop crop_only --batchSize 100 --no_html --gpu_ids 0,1,2,3 --self_gen --nThreads 4 --n_downsample_global 3 --k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan --outputs_dir [您的输出文件夹] --checkpoints_dir [您的检查点文件夹]

python train_domain_B.py --continue_train --training_dataset domain_B --name domainB_old_photos --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [您的数据文件夹] --no_instance --resize_or_crop crop_only --batchSize 120 --no_html --gpu_ids 0,1,2,3 --self_gen --nThreads 4 --n_downsample_global 3 --k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan --outputs_dir [您的输出文件夹] --checkpoints_dir [您的检查点文件夹]
```

注意：对于 `--name` 选项，请确保您的实验名称包含 "domainA" 或 "domainB"，这将用于选择不同的数据集。

### 3）训练域间映射网络

训练无划痕映射：

```
python train_mapping.py --use_v2_degradation --training_dataset mapping --use_vae_which_epoch 200 --continue_train --name mapping_quality --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [您的数据文件夹] --no_instance --resize_or_crop crop_only --batchSize 80 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [domainA_SR_old_photos的检查点] --load_pretrainB [domainB_old_photos的检查点] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [您的输出文件夹] --checkpoints_dir [您的检查点文件夹]
```

训练有划痕映射：

```
python train_mapping.py --no_TTUR --NL_res --random_hole --use_SN --correlation_renormalize --training_dataset mapping --NL_use_mask --NL_fusion_method combine --non_local Setting_42 --use_v2_degradation --use_vae_which_epoch 200 --continue_train --name mapping_scratch --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [您的数据文件夹] --no_instance --resize_or_crop crop_only --batchSize 36 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [domainA_SR_old_photos的检查点] --load_pretrainB [domainB_old_photos的检查点] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [您的输出文件夹] --checkpoints_dir [您的检查点文件夹] --irregular_mask [掩码文件的绝对路径]
```

训练有划痕映射（用于高分辨率输入的多尺度补丁注意力机制）：

```
python train_mapping.py --no_TTUR --NL_res --random_hole --use_SN --correlation_renormalize --training_dataset mapping --NL_use_mask --NL_fusion_method combine --non_local Setting_42 --use_v2_degradation --use_vae_which_epoch 200 --continue_train --name mapping_Patch_Attention --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [您的数据文件夹] --no_instance --resize_or_crop crop_only --batchSize 36 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [domainA_SR_old_photos的检查点] --load_pretrainB [domainB_old_photos的检查点] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [您的输出文件夹] --checkpoints_dir [您的检查点文件夹] --irregular_mask [掩码文件的绝对路径] --mapping_exp 1
```

## 引用

如果您觉得我们的工作对您的研究有帮助，请考虑引用以下论文 :)

```bibtex
@inproceedings{wan2020bringing,
title={Bringing Old Photos Back to Life},
author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={2747--2757},
year={2020}
}
```

```bibtex
@article{wan2020old,
  title={Old Photo Restoration via Deep Latent Space Translation},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  journal={arXiv preprint arXiv:2009.07047},
  year={2020}
}
```

如果您也对老照片/视频上色感兴趣，请参阅[这个项目](https://github.com/zhangmozhe/video-colorization)。

## 维护

本项目目前由万子瑜（Ziyu Wan）维护，仅供学术研究使用。如有任何问题，请随时联系 raywzy@gmail.com。

## 许可证

本仓库中的代码和预训练模型遵循 LICENSE 文件中指定的 MIT 许可证。我们使用自己标注的数据集来训练划痕检测模型。

本项目已采用 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。更多信息请参阅[行为准则常见问题](https://opensource.microsoft.com/codeofconduct/faq/)，或通过 [opencode@microsoft.com](mailto:opencode@microsoft.com) 联系我们咨询其他问题或意见。