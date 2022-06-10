# PPOCR-ONNX


## 简介

利用 onnxruntime 及 PaddleOCR 提供的模型, 对图片中的文字进行检测与识别.

### 使用模型

 - 文字检测: `ch_PP-OCRv3_det_infer`
 - 方向分类: `cls mobile v2`
 - 文字识别: `ch_PP-OCRv2_rec_infer`

## 参考

 - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
 - [手把手教你使用ONNXRunTime部署PP-OCR](https://aistudio.baidu.com/aistudio/projectdetail/1479970)
 - `ch_PP-OCRv3_det_infer` 及 `ch_PP-OCRv2_rec_infer` 模型来自 [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR)

## 安装

```bash
pip install ppocr-onnx
```

## 使用

```python
from ppocronnx.predict_system import TextSystem
import cv2

text_sys = TextSystem()

# 识别单行文本
res = text_sys.ocr_single_line(cv2.imread('single_line_text.png'))
print(res)

# 批量识别单行文本
res = text_sys.ocr_lines([cv2.imread('single_line_text.png')])
print(res[0])

# 检测并识别文本
img = cv2.imread('test.png')
res = text_sys.detect_and_ocr(img)
for boxed_result in res:
    print("{}, {:.3f}".format(boxed_result.ocr_text, boxed_result.score))
```
