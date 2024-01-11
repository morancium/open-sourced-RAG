'''This is created to test the OCR function and make the BBs of the OCRs'''
import pytesseract
import cv2
from pytesseract import Output
import os
from PIL import Image
from paddleocr import PaddleOCR,draw_ocr
import os
from pdf2image import convert_from_path

# from pdf2image import convert_from_path   

def pytessocr(image_path,save_path,config='--psm 3 --oem 3'):
    img = cv2.imread(image_path)
    d = pytesseract.image_to_data(img, output_type=Output.DICT,config=config)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0, 0), 1)
    cv2.imwrite(save_path, img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

# ocr = PaddleOCR(use_angle_cls=True, lang='en')
def ppocr(img_path,save_path):
    result = ocr.ocr(img_path, cls=True)
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='/root/work/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(save_path)


def text_from_pytess(image_path):
    image= cv2.imread(image_path)
    text = pytesseract.image_to_string(image,lang='eng',config='--psm 4 --oem 3')
    with open("sample.txt", 'w') as f:
        f.write(text)
    return text
    # print(text)

def conv_pdf_to_img(pdf_path,save_path):
    pages = convert_from_path(pdf_path, 500)
    #Saving pages in jpeg format
    count=0
    if len(pages)>1:
        for page in pages:
            count+=1
            page.save(save_path+"_"+str(count)+'.png', 'PNG')
            # print(count)
    else:
        for page in pages:
            page.save(save_path+'.png', 'PNG')


# conv_pdf_to_img("/root/work/Business_Analyst.pdf","hello")
# pytessocr('out0.png',"image_pt.png")
# ppocr('out0.png',"image.png")
# text_from_pytess('out0.png')
# print("Hellp")