from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

'''here we are going to define various prompts'''
JD='''compensation: Not disclosed
location: Gurugram,Haryana, Mumbai (All Areas)
About: udaanCapital is a digital lending platform and provides access to credit to empower Indian MSMEs by bringing lenders, brands and platforms together. We create working capital management products with intuitively embedding financing.
udaanCapital has facilitated disbursals of over 4,000 crore of working capital to over 2,500 MSME buyers in the last 24 months. These disbursals were enabled through partnership with Brand Partners via udaanCapital's Supply Chain Financing programme.

Role: Business Development Manager (Anchor Acquisition)- West & North Zone
Experience- 04-10 Years in enterprise sales. Experience in supply chain financing is a plus.
Job Objective: To be a part of the brand anchor acquisition team for scaling up the Supply Chain Finance book.

Key Responsibilities:
Prospecting and on-boarding new Anchors for Dealer Finance propositions and ensuring maximum onboarding and utilization of limits.
Building pipeline of brand anchors by identifying key markets
Networking with CXOs and other decision makers of brand anchors
Participation in industry forums to create visibility for udaanCapital
Implementation of various digital initiatives including new product development.
Regular engagement with anchors to spot red flags, build more inroads and scale the program.
Cohesive working with various internal stakeholders; Sales, Risk, Operations, Compliance, etc

Minimum Qualification: MBA or CA or Masters in Finance or Economics
Please reach us at chetna.setia@udaan.com
Industry Type: Travel & Tourism
Department: Sales & Business Development
Employment Type: Full Time, Permanent
Role Category: BD / Pre Sales'''


def run(jd=JD):
    base_prompt = f'''GPT4 User: Suppose you are a very well known and experienced linguist, given below the context below in between << >> you have to retrieve the important information like Budget, Notice period, No. of vacancies, Years of experience, required skills and qualifications. You have to return in the following format: [("entity 1” : "type of entity 1"), ... ].
If any of the above mentioned field are not there return “N/A” try to give a precise answer don't try to make up extra things, please do what is asked for
<< {jd} >>
<|end_of_turn|>GPT4 Assistant:'''
    return base_prompt

def query_prompt(context, query):
    base_prompt= f'''GPT4 User: Suppose you are a very well known and experienced linguist, given below the context below in between << >> along with you are also provided with the Questions in between <[ ]> which a user is asking from the context provided, please find the answer of the query from the given context only, if you DONT KNOW THE ANSWER JUST SAY "I DONT KNOW", else do not make up the answer!
    
    Context:
    << {context} >>
    
    Query:
    <[ {query} ]>
    <|end_of_turn|>GPT4 Assistant:'''
    return base_prompt


# print(jd)
# print("\n\n*** Generate:")
def model_inference(model_name_or_path,cv="",jd="",query="")->str:
    print("\n\n started!!")
    prompt_template=run(jd)
    if model_name_or_path=="openchat-3.5":
        model_name_or_path="TheBloke/openchat_3.5-GPTQ"
        revision="main"
    if model_name_or_path=="Mixtral":
        model_name_or_path="TheBloke/Mixtral-8x7B-v0.1-GPTQ"
        revision="main"
    if model_name_or_path=="vicuna-13b-v1.5-16k":
        model_name_or_path="TheBloke/vicuna-13B-v1.5-16K-GPTQ"
        revision="main"
    if model_name_or_path=="zephyr-7b-beta":
        model_name_or_path="TheBloke/zephyr-7B-beta-GPTQ"
        revision="main"
    
    if jd=="":
        context="\n".join(l for l in cv)
        context=query_prompt(context,query)
    else:
        context=prompt_template
    # To use a different branch, change revision
    # For example: revision="gptq-4bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    input_ids = tokenizer(context, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=1024)
    b = tokenizer.decode(output[0])
    idx1 = tokenizer.decode(output[0]).find("GPT4 Assistant:")
    idx2=max(tokenizer.decode(output[0]).find("</s>"),tokenizer.decode(output[0]).rfind("<|end_of_turn|>"))
    print("-------------------")
    b=b[idx1+15:idx2]
    # print(b,idx2)
    return b

# out=model_inference(jd=JD,model_name_or_path="zephyr-7b-beta")
# print(out)