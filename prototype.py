import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA initialization
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["allow_dangerous_deserialization"] = "True"
print(os.getcwd())
embedding_path="index.faiss"
print(f"Loading FAISS index from: {embedding_path}")
print("Version 11:03")
if not os.path.exists(embedding_path):
    print("File not found!")
#HF_KEY=os.getenv('Gated_Repo')
SAIA_KEY = "API-KEY"

#import spaces
import time
from typing import final
import asyncio

import torch
import gradio as gr
import threading
import re
import csv
import json
import gc
import multiprocessing
#import resource


from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.indexing import index
from langchain_core.vectorstores import VectorStore
from llama_index.core.node_parser import TextSplitter
#from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from llama_index.legacy.vector_stores import FaissVectorStore
from pycparser.ply.yacc import token
from ragatouille import RAGPretrainedModel

from langchain_text_splitters import MarkdownHeaderTextSplitter, CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy.testing.suite.test_reflection import metadata
from sympy.solvers.diophantine.diophantine import length
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from transformers import pipeline

#DEPR:from langchain.vectorstores import FAISS
import faiss
from langchain_community.vectorstores import FAISS
#DEPR: from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from huggingface_hub import login

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#login(token=HF_KEY)
vectorstore=None
rerankingModel=None
bm25_retriever=None
docstore=None

class BSIChatbot:
    embedding_model = None
    llmpipeline = None
    llmtokenizer = None
    vectorstore = None
    images = [None]


    llm_base_url = "https://chat-ai.academiccloud.de/v1"
    llm_remote_model = "qwen2.5-72b-instruct"
    llm_client = OpenAI(
        api_key = SAIA_KEY,
        base_url = llm_base_url
    )

    #llm_path = "meta-llama/Llama-3.2-3B-Instruct"
    word_and_embed_model_path = "H:\\Uni\\Master\\Masterarbeit\\Masterarbeit\\Models\\multilingual-e5-large-instruct"
    docs = "H:\\Uni\\Master\\Masterarbeit\\PrototypGrundschutzChatbot\docs"
    rerankModelPath="H:\\Uni\\Master\\Masterarbeit\\Masterarbeit\\Models\\ColBERTv1.0-german-mmarcoDE"
    embedPath="H:\\Uni\\Master\\Masterarbeit\\testSAIAApi\\docs\\_embeddings"

    def __init__(self):
        self.embedding_model = None


    def cleanResources(self):
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 / 1024} MB")
        multiprocessing.active_children()
        print("processes:")
        print(multiprocessing.active_children())

        for child in multiprocessing.active_children():
            child.terminate()
            child.join()

        torch.cuda.empty_cache()
        gc.collect()

    def initializeEmbeddingModel(self, new_embedding):
        global vectorstore
        RAW_KNOWLEDGE_BASE = []

        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.word_and_embed_model_path,
                multi_process=False,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},  # True = cosine similarity
            )

        dirList = os.listdir(self.docs)
        if (new_embedding==True):
            for doc in dirList:
                print(doc)
                if (".md" in doc):
                    ##doctxt = TextLoader(docs + "\\" + doc).load()
                    file = open(self.docs + "\\" + doc, 'r', encoding='utf-8')
                    doctxt = file.read()
                    RAW_KNOWLEDGE_BASE.append(LangchainDocument(page_content=doctxt, metadata={"source": doc}))
                    file.close()
                if (".txt" in doc):
                    file = open(self.docs + "\\" + doc, 'r', encoding='cp1252')
                    doctxt = file.read()
                    if doc.replace(".txt",".png") in dirList:
                        RAW_KNOWLEDGE_BASE.append(LangchainDocument(page_content=doctxt, metadata={"source": doc.replace(".txt",".png")}))
                    if doc.replace(".txt",".jpg") in dirList:
                        RAW_KNOWLEDGE_BASE.append(LangchainDocument(page_content=doctxt, metadata={"source": doc.replace(".txt",".jpg")}))
                    file.close()

            # Chunking starts here

            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=True
            )

            tokenizer = AutoTokenizer.from_pretrained(self.word_and_embed_model_path)

            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=512,  # max number of words in a chunk
                chunk_overlap=0,  # number of characters to overlap between chunks
                add_start_index=True,  # includes chunk's start index in metadata
                strip_whitespace=True,  # strips whitespace from the start and end of every document
            )

            docs_processed = []
            for doc in RAW_KNOWLEDGE_BASE:
                doc_cache = markdown_splitter.split_text(doc.page_content)
                doc_cache = text_splitter.split_documents(doc_cache)

                for chunk in doc_cache:
                    chunk.metadata.update({"source": doc.metadata['source']})

                docs_processed += doc_cache



            # Make sure the maximum length is below embedding size
            lengths = [len(s.page_content) for s in docs_processed]
            print(max(lengths))

            start = time.time()

            vectorstore = FAISS.from_documents(docs_processed, self.embedding_model, distance_strategy=DistanceStrategy.COSINE)

            vectorstore.save_local(self.embedPath)
            end = time.time()
        else:
            start = time.time()
            if vectorstore is None:
                print("Checkpoint: FAISS Vectorstore initialized...")
                vectorstore = FAISS.load_local(self.embedPath, self.embedding_model, allow_dangerous_deserialization=True)
            end = time.time()


    def retrieveSimiliarEmbedding(self, query):
        global vectorstore
        print("Retrieving Embeddings...")
        start = time.time()
        query = f"Instruct: Given a search query, retrieve the relevant passages that answer the query\nQuery:{query}"

        retrieved_chunks = vectorstore.similarity_search(query=query, k=30)

        end = time.time()

        return retrieved_chunks

    def retrieveDocFromFaiss(self):
        global vectorstore
        global docstore
        all_documents = []

        # Iteriere über alle IDs im index_to_docstore_id
        if docstore is None:
            docstore = vectorstore.docstore._dict.values()

        for entry in docstore:
            all_documents.append(entry)

        return all_documents

    def queryLLM(self,query):
        return(self.llmpipeline(query)[0]["generated_text"])

    def initializeRerankingModel(self):
        global rerankingModel
        if rerankingModel is None:
            print("Checkpoint: Reranker initialized...")
            rerankingModel = RAGPretrainedModel.from_pretrained(self.rerankModelPath)


    def retrieval(self, query, rerankingStep, hybridSearch):
        global vectorstore
        global bm25_retriever
        global rerankingModel
        if hybridSearch == True:
            allDocs = self.retrieveDocFromFaiss()
            if bm25_retriever is None:
                bm25_retriever = BM25Retriever.from_documents(allDocs)
            #TODO!
            retriever_k=25
            bm25_retriever.k= retriever_k
            vectordb = vectorstore.as_retriever(search_kwargs={"k":retriever_k})
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vectordb], weights=[0.5, 0.5])
            retrieved_chunks = ensemble_retriever.invoke(query)
        else:
            retrieved_chunks = self.retrieveSimiliarEmbedding(query)
        retrieved_chunks_text = []
        # TODO Irgendwas stimmt hier mit den Listen nicht
        for chunk in retrieved_chunks:
            # TODO Hier noch was smarteres Überlegen für alle Header
            if "Header 1" in chunk.metadata.keys():
                retrieved_chunks_text.append(
                    f"The Document is: '{chunk.metadata['source']}'\nHeader of the Section is: '{chunk.metadata['Header 1']}' and Content of it:{chunk.page_content}")
            else:
                retrieved_chunks_text.append(
                    f"The Document is: '{chunk.metadata['source']}'\nImage Description is: ':{chunk.page_content}")
        i = 1
        for chunk in retrieved_chunks_text:
            i = i + 1

        if rerankingStep == True:
            if rerankingModel is None:
                self.initializeRerankingModel()
            print("Starting Reranking Chunks...")
            with torch.no_grad():
                print("reranking chunks (reverse)..")
                retrieved_chunks_text = rerankingModel.rerank(query, retrieved_chunks_text, k=20)

            retrieved_chunks_text = [chunk["content"] for chunk in reversed(retrieved_chunks_text)]


            i = 1
            for chunk in retrieved_chunks_text:
                i = i + 1

        context = "\nExtracted documents:\n"
        context += "".join([doc for i, doc in enumerate(retrieved_chunks_text)])

        return query, context

    def queryRemoteLLM(self, systemPrompt, query, summary):
        if summary != True:
            chat_completion = self.llm_client.chat.completions.create(
                messages=[{"role": "system", "content": systemPrompt},
                          {"role": "user", "content": "Step-Back Frage, die neu gestellt werden soll: " + query}],
                model=self.llm_remote_model,
            )
        if summary == True:
            chat_completion = self.llm_client.chat.completions.create(
                messages=[{"role": "system", "content": systemPrompt},
                          {"role": "user", "content": query}],
                model=self.llm_remote_model,
            )
        return chat_completion.choices[0].message.content

    def stepBackPrompt(self, query):
        systemPrompt = """
        Sie sind ein Experte für den IT-Grundschutz des BSI. 
        Ihre Aufgabe ist es, eine Frage neu zu formulieren und sie in eine
        Stepback-Frage umzuformulieren, die nach einem Grundkonzept der Begrifflichkeit fragt. 

        Hier sind ein paar Beispiele:
        Ursprüngliche Frage: Welche Bausteine werden auf einen Webserver angewendet?
        Stepback-Frage: Wie sind Bausteine laut dem IT-Grundschutz anzuwenden?

        Ursprüngliche Frage: Was sind Beispiele für die Gefährdung Eindringen in IT-Systeme?
        Stepback-Frage: Was sind Gefährdungen im IT-Grundschutz?

        Ursprüngliche Frage: Welche Inhalte enthält der Standard 200-1?
        Stepback Frage: Welche Standards gibt es im IT-Grundschutz?
        """
        stepBackQuery = self.queryRemoteLLM(systemPrompt, query, False)
        return stepBackQuery

    def ragPromptNew(self, query, rerankingStep, history, stepBackPrompt, returnContext):
        global rerankingModel
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": """You are a helpful Chatbot for the BSI IT-Grundschutz. Using the information contained in the context,
                        give a comprehensive answer to the question.
                        Respond only to the question asked, response should be concise and relevant but also give some context to the question. 
                        Provide the source document when relevant for the understanding.
                        If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """Context:
                        {context}
                        ---
                        Chat-History:
                        {history}
                        ---
                        Now here is the question you need to answer.

                        Question: {question}""",
            },
        ]
        # RAG_PROMPT_TEMPLATE = self.llmtokenizer.apply_chat_template(
        #    prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        # )

        # Alles außer letzte Useranfrage, Normaler Query
        query, context = self.retrieval(query, rerankingStep, True)

        if stepBackPrompt == True:
            stepBackQuery = self.stepBackPrompt(query)
            print("DBG stepBackQuery:" + stepBackQuery)
            stepBackQuery, stepBackContext = self.retrieval(stepBackQuery, rerankingStep, True)
            #newprint("DBG stepBackContext:" + stepBackContext)
            sysPrompt = """
            You are an helpful Chatbot for the BSI IT-Grundschutz. Using the information contained in the context,
                give a comprehensive answer to the question.
                Respond only to the question asked, response should be concise and relevant but also give some context to the question. 
                Provide the source document when relevant for the understanding.
                If the answer cannot be deduced from the context, do not give an answer.
                """
            stepBackAnswer = self.queryRemoteLLM(sysPrompt, stepBackQuery, True)
            #newprint("DBG stepBackAnswer:" + stepBackAnswer)
            context += "Übergreifende Frage:" + stepBackQuery + "Übergreifender Context:" + stepBackAnswer

        #def queryRemoteLLM(self, systemPrompt, query, summary):

        if (history != None):
            prompt_in_chat_format[-1]["content"] = prompt_in_chat_format[-1]["content"].format(
                question=query, context=context, history=history[:-1]
            )
        else:
            prompt_in_chat_format[-1]["content"] = prompt_in_chat_format[-1]["content"].format(
                question=query, context=context, history="None"
            )
        final_prompt = prompt_in_chat_format

        # final_prompt = prompt_in_chat_format[-1]["content"].format(
        #    question=query, context=context, history=history[:-1]
        # )

        print(f"DBG: Final-Query:\n{final_prompt}")
        pattern = r"Filename:(.*?);"
        last_value = final_prompt[-1]["content"]

        match = re.findall(pattern, last_value)
        self.images = match

        if (returnContext == False):
            stream = self.llm_client.chat.completions.create(
                messages=final_prompt,
                model=self.llm_remote_model,
                stream=True
            )
            return stream

        else:
            answer = self.llm_client.chat.completions.create(
                messages=final_prompt,
                model=self.llm_remote_model,
                stream=False
            )
            self.cleanResources()
            return answer, context

    def returnImages(self):
        imageList = []
        for image in self.images:
            imageList.append(f"{self.docs}\\{image}")
        return imageList

    def launchGr(self):
        gr.Interface.from_pipeline(self.llmpipeline).launch()

    def generateEvalDataset(self):
        filepath = "H:\\Uni\\Master\\Masterarbeit\\testSAIAApi\\docs\\_eval\\BSI_Lektion_Ground_Truth.CSV"
        with open(filepath, mode='r', encoding="latin1", errors="replace") as file:
            # Create a CSV reader with DictReader
            csv_reader = csv.DictReader(file, delimiter="|")

            # Initialize an empty list to store the dictionaries
            data_list = []

            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Append each row (as a dictionary) to the list
                data_list.append(row)

        # Print the list of dictionaries
        for data in data_list:
            data["Context"] = None
            data["Answer"] = None

        print(data_list)

        i=1
        #for data in data_list[:3]:
        print("starting to generate evaldataset..")
        for data in data_list:
            print("Eval Entry no:")
            print(i)
            print("GPU Memory Allocated:")
            print(torch.cuda.memory_allocated()/1024/1024/1024)
            print("frage:")
            print(data["Frage"])
            #def ragPromptNew(self, query, rerankingStep, history, stepBackPrompt)
            try:
                #print(self.using("PreRag"))
                data["Answer"],data["Context"]  = self.ragPromptNew(data["Frage"],True,None,True, True)
                #print(self.using("AfterRag"))
                data["Answer"]=data["Answer"].choices[0].message.content
            except Exception as e:
                print(f"Fehler bei Eintrag {i}: {e}")

            print("DBG: storing Answer")
            print(data["Answer"][:20])
            print("in")
            print(data["Frage"])
            print(data["Frage_index"])
            print(data["Lektion"])
            #print(data)
            i=i+1
            with open('H:\\Uni\\Master\\Masterarbeit\\testSAIAApi\\docs\\_eval\\dataset.json', 'a') as fout:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")


        # Print full response as JSON
        # print(chat_completion)

if __name__ == '__main__':


    eval = False
    renewEmbeddings = False
    reranking = False
    stepBackEnable = False

    bot = BSIChatbot()
    bot.initializeEmbeddingModel(renewEmbeddings)
    if reranking == True:
        bot.initializeRerankingModel()

    if (eval==True):
        bot.generateEvalDataset()
    #bot.launchGr()

    with gr.Blocks() as demo:
        with gr.Row() as row:
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(type="messages")
                    msg = gr.Textbox()
                    clear = gr.Button("Clear")
                    reset = gr.Button("Reset")
                with gr.Column(scale=1):  # Bildergalerie
                    gallery = gr.Gallery(label="Bildergalerie",elem_id="gallery")

        def user(user_message, history: list):
            return "", history + [{"role": "user", "content": user_message}]


        def returnImages():
            # Hier  Bildpfade und in gr.Image-Objekte umwandeln
            image_paths = bot.returnImages()
            print(f"returning images: {image_paths}")
            return image_paths

        def gradiobot(history: list):
            start = time.time()
            print(f"DBG: ragQuery hist -1:{history[-1].get('content')}")
            print(f"DBG: ragQuery hist 0:{history[0].get('content')}")
            print(f"DBG: fullHistory: {history}" )
            #bot_response = bot.ragPromptRemote(history[-1].get('content'), reranking, history)
            bot_response = bot.ragPromptNew(history[-1].get('content'), reranking, history, stepBackEnable, False)
            history.append({"role": "assistant", "content": ""})

            image_gallery = returnImages()

            for token in bot_response:
                if token.choices and len(token.choices) > 0:
                    if token.choices[0].delta.content != "":
                        history[-1]['content'] += token.choices[0].delta.content
                        yield history, image_gallery
            end = time.time()
            print("End2End Query took", end - start, "seconds!")

        def resetHistory():
            return []

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            gradiobot, inputs=[chatbot], outputs=[chatbot, gallery]
        )


        clear.click(lambda: None, None, chatbot, queue=False)
        reset.click(resetHistory, outputs=chatbot, queue=False)
    demo.css = """
        #gallery {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            height: 400px;
            overflow: auto;
        }
    """
    demo.launch()
