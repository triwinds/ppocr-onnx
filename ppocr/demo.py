from ppocr.predict_system import TextSystem
import cv2
import logging
import sys
from PIL import Image
from ppocr.utility import draw_ocr_box_txt


def main():
    text_sys = TextSystem()
    img = cv2.imread('test.png')
    img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    res = text_sys.detect_and_ocr(img)
    for boxed_result in res:
        print("{}, {:.3f}".format(boxed_result.ocr_text, boxed_result.score))
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = [boxed_result.box for boxed_result in res]
    txts = [boxed_result.ocr_text for boxed_result in res]
    scores = [boxed_result.score for boxed_result in res]

    draw_img = draw_ocr_box_txt(
        image,
        boxes,
        txts,
        scores,
        drop_score=0.3,
        font_path='WeiRuanYaHei-1.ttf')
    cv2.imshow('test', draw_img[:, :, ::-1])
    cv2.waitKey()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
