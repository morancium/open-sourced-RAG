# from library import *
import glob
import argparse
from ocr import *
from retrival import insert_db_text, retrive_topK
from inference import model_inference
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source",help="path to the source",nargs="+",type=str)
ap.add_argument("-q", "--query",help="Write your Query",nargs='+',type=str)

args = ap.parse_args()
source=args.source
query=args.query

image_ext=['png','jpg','jpeg']

for path in source:
    ext=path.split(".")[-1]
    text=""
    if(ext in image_ext):
        print("converting Image to text")
        text=text_from_pytess(path)
        print("Almost there...")
        pass
    if(ext=="pdf"):
        print("converting PDF into Text")
        files = glob.glob('/root/work/temp/*')
        for f in files:
            os.remove(f)
        conv_pdf_to_img(path,"temp/img")
        print("Almost there...")
        for img in sorted(os.listdir("temp")):
            # print("/root/work/temp/"+img)
            temp=text_from_pytess("/root/work/temp/"+img)
            text+="\n"+temp
        pass
    if(ext=="txt"):
        print("Reading the txt file")
        with open (path, "r") as myfile:
            text = myfile.read()
        pass
    # print(ext)
# print(text)
insert_db_text(text)
results=retrive_topK(query)
for i,qes in enumerate(query):
    print("Query: \n",qes)
    print(results[i])
    answer=model_inference(model_name_or_path="zephyr-7b-beta",cv=results[i],query=qes)
    print("\n\n\n")
    print("Query: \n",qes)
    print(answer)
    pass
# print(source)
# print(query)