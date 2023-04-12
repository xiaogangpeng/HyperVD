
This is a PyTorch implementation for ***Learning Weakly Supervised Audio-Visual Violence Detection in Hyperbolic Space***. The original paper can be found [here](https://xiaogangpeng.github.io/). Please feel free to contact us via email if you have any further questions or inquiries.
# Abstract 
In recent years, the field of weakly supervised audio-visual violence detection has gained substantial attention. The goal of this task is to identify multimodal violent snippets based on the video-level label. Despite the progress made, traditional Euclidean neural networks used in previous methods still face challenges in capturing discriminative representations. To overcome this limitation, we propose HyperVD, a novel framework that learns snippet embeddings in hyperbolic space to improve model discrimination. Specifically, our framework comprises a detour fusion module for multimodal fusion, which effectively alleviates modality inconsistency, and two branches of fully hyperbolic graph convolutional networks, which excavate feature similarities and temporal relationships among snippets in hyperbolic space. Extensive experiments on the XD-Violence benchmark demonstrate that our method outperforms the state-of-the-art methods by a sizable margin. 


## Training Stage
- Download the extracted I3D features of XD-Violence dataset from [here](https://roc-ng.github.io/XD-Violence/).
- Change the file paths of ```make_list.py``` in the list folder to generate the training and test list.
-  The hyperparameters are saved in ```option.py```, where we keep default settings as mentioned in our paper.
- Run the following command for training:
```
python main.py
```
## Test Stage
- Change the checkpoint path of ```infer.py```.
- Run the following command for test:
```
python infer.py
```

## Acknowledgements
The implementation mainly references the repositories of [XDVioDet](https://github.com/Roc-Ng/XDVioDet) and [fully-hyperbolic-nn
](https://github.com/chenweize1998/fully-hyperbolic-nn). We greatly appreciate their excellent contribution.


If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{xx,
  title={Learning Weakly Supervised Audio-Visual Violence Detection in Hyperbolic Space},
  author={xx},
  journal={arXiv preprint arXiv:xx},
  year={2023}
}
```
