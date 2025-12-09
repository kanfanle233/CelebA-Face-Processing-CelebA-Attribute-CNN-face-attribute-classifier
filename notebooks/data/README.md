# CelebA Dataset Instructions

The CelebA dataset is NOT included in this repository due to its size 
and licensing restrictions.

To use this project, download CelebA from:

🔗 https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html  
or  
🔗 https://drive.google.com/drive/folders/0B7EVK8r0v71pQ-1tLS0tNkZvcFo

## Place the files like this:

data/
└── archive/
    ├── list_attr_celeba.csv
    ├── list_bbox_celeba.csv
    ├── list_eval_partition.csv
    ├── list_landmarks_align_celeba.csv
    └── img_align_celeba/           ← contains ~200k face images (NOT uploaded)

## Notes
- The `.csv` annotation files are required for preprocessing.
- The `img_align_celeba/` folder contains all face images and must be downloaded manually.
- This repository only includes the code needed to process and train on the dataset.

