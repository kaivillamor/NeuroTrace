# NeuroTrace

This project aims to use a U-Net model in PyTorch to predict pixel-level masks that outline tumor regions in brain MRI images.

**Metric**
Dice Score - measures how much the predicted tumor mask overlaps with the actual tumor mask. Ranges from 0 (being no overlap) to 1 (perfect overlap). Used instead of plain accuracy because the majority of pixels in any image are non-tumor, which would make accuracy misleading.

**Data**
LGG Brain MRI Segmentation Dataset - https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

**How to Run**
Open the notebook in Google Colab and run the cells top to bottom.

**Dependencies**
- PyTorch
- Albumentations
- OpenCV
- Matplotlib
