from angle_emb import AnglE, Prompts
from chromadb.utils import embedding_functions
import chromadb
import langchain
chroma_client = chromadb.Client()
import shutil

# client = chromadb.PersistentClient(path="/root/work/db0")
# JD_info = client.get_or_create_collection(name="JD_info",metadata={"hnsw:space": "cosine"})

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
angle.set_prompt(prompt=Prompts.C)

# Please upload your samples in the directory!!
paths_to_text=["/root/work/sample/sample1.txt",
               "/root/work/sample/sample2.txt",
               "/root/work/sample/sample3.txt",
               "/root/work/sample/sample4.txt",
               "/root/work/sample/sample5.txt",
               "/root/work/sample/sample6.txt",
               "/root/work/sample/sample7.txt",
               "/root/work/sample/sample8.txt",
               "/root/work/sample/sample9.txt",
               "/root/work/sample/sample10.txt",]

def insert_db(text="",paths_to_text=paths_to_text):
    for i in range(len(paths_to_text)):
        with open (paths_to_text[i], "r",errors='ignore') as myfile:
            data = myfile.read()
        text = data

        # print(text)
        print("Upserting...")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(

            chunk_size = 512,
            chunk_overlap  = 100
        )

        docs = text_splitter.create_documents([text])
        chunked=[chunk.page_content for chunk in docs]

        # print(len(docs))
        # print(len(chunked))
        # vecs = angle.encode([{'text': 'hello world1'}, {'text': 'hello world2'}], to_numpy=True)
        inp=[{'text': text} for text in chunked]
        vecs = angle.encode(inp, to_numpy=True)
        # print(len(vecs))
        ids=["jd_"+str(i+1)+"_"+str(j+1) for j in range(len(chunked))]
        print(ids)
        JD_info.add(
            embeddings = vecs,
            documents = chunked,
            ids = ids)


def insert_db_text(text):
    shutil.rmtree("/root/work/db0")
    client = chromadb.PersistentClient(path="/root/work/db0")
    JD_info = client.get_or_create_collection(name="JD_info",metadata={"hnsw:space": "cosine"})

    # angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    # angle.set_prompt(prompt=Prompts.C)
    print("Upserting...")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size = 512,
        chunk_overlap  = 100
    )

    docs = text_splitter.create_documents([text])
    chunked=[chunk.page_content for chunk in docs]
    inp=[{'text': text} for text in chunked]
    vecs = angle.encode(inp, to_numpy=True)
    # print(len(vecs))
    ids=[str(j+1) for j in range(len(chunked))]
    print(ids)
    JD_info.add(
        embeddings = vecs,
        documents = chunked,
        ids = ids)


def retrive_topK(query=[],topk=4):
    client = chromadb.PersistentClient(path="/root/work/db0")
    JD_info = client.get_or_create_collection(name="JD_info",metadata={"hnsw:space": "cosine"})
    inp=[{'text': text} for text in query]
    query_embeddings = angle.encode(inp, to_numpy=True)
    results = JD_info.query(
        query_embeddings=query_embeddings,
        n_results=topk
    )
    return results['documents']

'''
# insert_db()

# Add your Query
query_texts=["tell me about LTIMindtree","Inspire Brands"]
inp=[{'text': text} for text in query_texts]
query_embeddings = angle.encode(inp, to_numpy=True)
# print(query_embeddings)

results = JD_info.query(
    query_embeddings=query_embeddings,
    n_results=5
)
# Cosine distance is the distance between two points in high dimensional space. It is defined as 1 - Similarity(A, B).
print(results['documents'][0])
print('\n\n\n')
print(results['documents'][1])
'''