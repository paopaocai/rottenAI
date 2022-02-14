# rottenAI

## Introduction
Our project have 4 parts. Each part should be set up and run individually. 
1) CNN Model to Classify Rotten and Fresh Banana Images
2) Non NN to Classify Rotten and Fresh Banana Images
3) YOLO v3 to detect Rotten and Fresh Bananas
4) YOLO v3 to detect Rotten and Fresh Part of Different Kinds of Fruits



## Model Setup

### Part 1: CNN Model to Classify Rotten and Fresh Banana Images
- Code
  - main.py
- Data
  - Data

### Part 2: Non NN to Classify Rotten and Fresh Banana Images
- Code
  - model/NonNN_RottenDetect.ipynb
- Data
  - model/Data

### Part 3: YOLO v3 to detect Rotten and Fresh Bananas
- Code
  - model/YOLOv3_banana.ipynb
  - These part need to download network of Darknet-53. Run the following code first.
    ```python
    !wget https://pjreddie.com/media/files/darknet53.conv.74
    ```
  - If you want to test our model. Please download our trained weights first.
    ```
    https://drive.google.com/file/d/12CrathFncIFFY7uJrDU3qWOAv2GmmI44/view?usp=sharing
    ```
- Data
  - model/darknet-banana/VOCdevkit/VOC2007
  

### Part 4: YOLO v3 to detect Rotten and Fresh Part of Different Kinds of Fruits
- Code
  - model/YOLOv3_rotten_part.ipynb
  - These part need to download network of Darknet-53. Run the following code first.
    ```python
    !wget https://pjreddie.com/media/files/darknet53.conv.74
    ```
  - If you want to test our model. Please download our trained weights first.(Different from part 3)
    ```
    https://drive.google.com/file/d/1ACC8ETSiflpyO6eU3cDn7oUj-BLuKqjx/view?usp=sharing
    ```
- Data
  - model/darknet-rotten-part/VOCdevkit/VOC2007



