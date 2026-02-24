# fluPictureProcess

> A chip decoding and fluorescent image processing software for third-generation full-length spatial transcriptomics.
> 三代全长空间转录组芯片解码与荧光图像处理软件。

## 模块说明 (Modules Overview)

本项目包含以下核心处理模块：

* **`ChipRegionSearch.py`**：自动从图片中识别芯片区域。
* **`FluorescentBase.py`**：基础荧光处理模块。
* **`HeuristicBeadSearch.py`**：通过启发式算法进行 bead 荧光信号点识别。
* **`ImageProcess.py`**：图片基础处理模块（包含二值化、灰度处理等功能）。
* **`MicroBeadMask.py`**：通过标准化 bead mask 进行荧光信号识别，并将结果保存在文件中（支持对整个芯片的全部子图进行逐一识别）。
* **`MicroBeadMaskDist.py`**：标准化 bead mask 分布处理模块。
* **`MicroBeadOutMask.py`**：标准化 bead mask 输出模块。
* **`MRXSBase.py`**：分层图片处理基础模块。
* **`MRXSRegionExtract.py`**：从分层图片中提取指定区域的图片信息。
* **`SplitFluImg.py`**：将原始荧光图片分割成一张张小的子图，方便后期荧光信号识别处理。

---
*Developed by lbs-lab*
