# Human Mask Detection with PyTorch

- 每部分文件作用描述（以codes为根文件夹，下同）：
    - data/：存放相关非代码文件，包括：
        - AIZOO数据库（./data/AIZOO/train/1_HandShanking...）（下载地址略）
        - 最优模型文件（checkpoint_zfr.pt，https://cloud.tsinghua.edu.cn/f/21abf4e68b814463ae29/）
    - models/：四个文件分别实现了四个模型
    - test-images/：10张测试图片
    - dataset.py：定义预处理函数、定义数据集类、定义transforms函数（包括数据增广）
    - engine.py：定义训练一个epoch函数、测试一个数据集函数、检测一张图片函数
    - main.ipynb：包含数据可视化以及四个模型的实验结果
    - metric.py：定义mAP和precision-recall计算函数
    - reproduce.py：运行时判断data文件夹内是否有checkpoint_zfr.pt，如无则开始预处理数据并训练Faster RCNN（1个epoch，详见代码第40行注释），如有则直接加载，之后开始检测10张测试图片。
    - utils.py：包含一些辅助函数
- 运行环境：Linux CentOS 7; Anaconda/Miniconda最新版（下载地址略）
    - conda create -n zfr python=3.7.7 pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
    - conda activate zfr
    - conda install lxml matplotlib pandas
- 下载数据、整理数据的命令：详见data文件夹描述，下载之后unzip并mv即可。
- 训练最优模型命令：
    - python reproduce.py
    - 详见reproduce.py文件描述
- 测试10张图片命令：同上
- 用提交的最优模型测试的命令：同上
    - 预期的结果：在test-images下生成10张annoted前缀的图像，包含检测框
    

