# here I will write the prompts needed for the retrievial of important fields
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pytesseract import Output
import pytesseract
import cv2, os,re
import json
import time



base_path="/root/work/open-sourced-RAG/"



model_name_or_path="TheBloke/vicuna-13B-v1.5-16K-GPTQ"


print("\n\n Warming up the model!! \n\n")
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


# reading from the photos
def text_from_pytess(image_path):
    image= cv2.imread(image_path)
    text = pytesseract.image_to_string(image,lang='eng',config='--psm 4 --oem 3')
    # with open("sample.txt", 'w') as f:
    #     f.write(text)
    return text


# with open ("/root/open-sourced-RAG/sample.txt", "r") as myfile:
#             text = myfile.read()
text=""
jd=text
# /root/work/open-sourced-RAG/resume_photos
out={}
for file in os.listdir(base_path+"resume_photos")[:5]:
    path=base_path+"resume_photos/"+file
    text=text_from_pytess(path)
    
    prompt1=""" 
    USER: 
    You're an expert ai, ml model. you give output in JSON format only. You as a model understand each and every line of the Job Description and company details like an HR and Sales head because the model training was done by transferring the expert knowledge of hr & Sales head person.
    Important task : Under each parameter, give the details from the given Job Description.
    Extract the following parameters from Job description and output in json format:
    "job_role" : " (Identify and extract the job title or position name)"
    "company_name" : "(Identify and extract the company name from the company details)"
    "total_experience" : "(Identify total years of experience required from job description)"
    "salary" : "(Identify and extract salary range, amount or stipend information from the document, it could be a range of number in various currencies)"
    "responsibilities" : "(Identify and extract information about the responsibilities a person has to have while joining the company)"
    "relevent_experience" : "(Identify and extract years of experience from job description relevant to the job role)"
    "required_skills" : "(Identify and extract required technologies or skills from job description)"
    "qualification" : "(Identify and extract minimum qualification related to EDUCATION required for the person applying for the role. For example Bachelor's)"
    "location" : "(Identify and extract location of the job role from the job description)"
    "unknown" : "(Identify and extract any other information from job description not mentioned in the above parameters)"
    Give the answer for each parameter in separate line. If the parameter is not present in the job description, then answer as "".
    return the output in json format such as parameter : value
    the job description is enclosed in << >>
    the job discription is as follows: <<"""
    
    prompt2=""">>

    
    please remember to give output in JSON format
    <|end_of_turn|>
    ASSISTANT:
    """
    final_prompt=prompt1+text+prompt2
    # print(final_prompt)
    start = time.time()
    input_ids = tokenizer(final_prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.001, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=1024)
    b = tokenizer.decode(output[0])
    
    # print(b)
    
    # b = re.findall(r"{.*?}", b)
    
    idx1 = tokenizer.decode(output[0]).rfind("ASSISTANT")
    idx2=tokenizer.decode(output[0]).rfind("</s>")
    end = time.time()
    print("---------------------------------------")
    print(file)
    b=b[idx1+10:idx2]
    b=b.replace("\_","_")
    print(b)
    # break
    json_object = json.loads(b)
    json_object["execution_time"] = str((end-start) * 10**3) + "ms"
    out[file]=json_object
    with open('vicuna_GPTQ.json', 'w') as outfile:
        json.dump(out, outfile)
