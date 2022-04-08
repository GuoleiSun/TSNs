# TSNs
Code for ICCV21 paper: Task Switching Network for Multi-task Learning

### Abstract
We introduce Task Switching Networks (TSNs), a task-conditioned architecture with a single unified encoder/decoder for efficient multi-task learning. Multiple tasks are performed by switching between them, performing one task at a time. TSNs have a constant number of parameters irrespective of the number of tasks. This scalable yet conceptually simple approach circumvents the overhead and intricacy of task-specific network components in existing works. In fact, we demonstrate for the first time that multi-tasking can be performed with a single task-conditioned decoder. We achieve this by learning task-specific conditioning parameters through a jointly trained task embedding network, encouraging constructive interaction between tasks. Experiments validate the effectiveness of our approach, achieving state-of-the-art results on two challenging multi-task benchmarks, PASCAL-Context and NYUD. Our analysis of the learned task embeddings further indicates a connection to task relationships studied in the recent literature.

### Results
Performance on PASCAL Context (ResNet-18 backbone) is as follows:

|  Experiment   | Edge Detection (F) | Semantic Segmentation (mIoU) | Human Parts (mIoU) | Surface Normals (mErr) | Saliency (mIoU)| Average Drop (%)|
| ------------  | ------------------ | ---------------------------- | ------------------ | ---------------------- | -------------- | --------------- |
|  Single Task  |        71.3        |             64.3             |        55.5        |            16.3        |       62.9     |          -      |
|      TSNs     |        70.6        |             64.2             |        55.0        |            16.3        |       63.3     |         0.30    |

###  Installation:
The following packages are required:
```
pytorch=1.7.1
scikit-image=0.17.2
pillow=8.3.1
opencv-python=4.4.0.46 
pycocotools=2.0.2
albumentations=1.0.0
```
Some other common packages are also needed.

### Trained models
1. Download this repo and go ino the project.

2. Go into `./fblib/util/`, and change `db_root` in `mypath.py` to be the data root you prefer. When running the code, the dataset will be automatically downloaded into this path if no datasets are found.

3. Download the single-task models from [here](https://drive.google.com/drive/folders/1qK1K07upTfWrQxLB2BpzQqoDwnLedPgE?usp=sharing) and TSNs model from [here](https://drive.google.com/drive/folders/1obX7rSvtPx27k4_GJ57DdHkeiv8V1BjK?usp=sharing).

4. For single tasks, run
```
# Take task of parts segmentation as an example
python test-single-task.py --batch_size 40 --ckdir /path/checkpoints/folder/ --best 129 --task parts
```
The results of tasks except edge detection will be printed out after executing the above script. However, the predictions of edge detection need to be evaluated by [seism repository](https://github.com/jponttuset/seism) (evaluation code is written in MATLAB and it is required to have it installed).

5. For TSNs, run
```
python test.py --batch_size 50 --ckdir /path/checkpoints/folder/ --best 129 
```

### Citation
If you find this repo helpful, please consider citing
```
@inproceedings{sun2021task,
  title={Task Switching Network for Multi-Task Learning},
  author={Sun, Guolei and Probst, Thomas and Paudel, Danda Pani and Popovi{\'c}, Nikola and Kanakis, Menelaos and Patel, Jagruti and Dai, Dengxin and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8291--8300},
  year={2021}
}
```
