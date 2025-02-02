# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
import openai
import time
import pandas

import os
from ragas import evaluate
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from langchain_openai import ChatOpenAI
from ragas.metrics import faithfulness, answer_correctness
from ragas.run_config import RunConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper

# Press the green button in the gutter to run the script.
import csv
import json
import os

def returnAnsw(question):
    return ("Antwort auf frage!" + question)

def returnCont():
    return "Context!"

if __name__ == '__main__':
    from openai import OpenAI

    # API configuration
    api_key = 'GPT-4oKey' #openai
    base_url = "https://chat-ai.academiccloud.de/v1"
    model = "meta-llama-3.1-70b-instruct"  # Choose any available model

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    
    
    filepath = "H:\\Uni\\BSI_Lektion_Ground_Truth.CSV"
    with open(filepath, mode='r') as file:
        
        csv_reader = csv.DictReader(file, delimiter="|")

        data_list = []
        for row in csv_reader:
            data_list.append(row)


    for data in data_list[:3]:
        print(data)
        data["Context"]=None
        data["Answer"] = None
        print(data)


    for data in data_list[:3]:
        data["Answer"]=returnAnsw(data["Frage"])
        data["Context"] = returnCont()
        print (data)
        print(data["Frage"])

    with open('H:\\Uni\\outputfile', 'w') as fout:
        json.dump(data_list, fout)
        


    def export_quiz_to_csv(data, output_file):
        try:
            with open(output_file, mode='w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file, delimiter='|')

                writer.writerow(['Frage', 'Antwortmöglichkeiten', 'Korrekt'])

                for section, questions in data['quiz'].items():
                    for qid, details in questions.items():
                        frage = details['question']
                        antworten = " | ".join([f"{key}: {value}" for key, value in details['answers'].items()])
                        korrekt = ", ".join(details['right'])

                        writer.writerow([frage, antworten, korrekt])
            print(f'Die CSV-Datei wurde erstellt: {output_file}')
        except Exception as e:
            print(f'Fehler beim Schreiben der Datei: {e}')

    def loadJSON():
        dataset = "H:\\Uni\\Master\\Masterarbeit\\testSAIAApi\\docs\\_eval\\dataset.json"

        if not os.path.exists(dataset):
            print(f"Fehler: Die Datei {dataset} existiert nicht.")
        else:
            try:
                with open(dataset, encoding='latin1') as f:
                    content = f.read().strip()
                    if not content:
                        raise ValueError("Die Datei ist leer.")
                    dataset = json.loads(content)

                data_samples = {
                    'question': [],
                    'answer': [],
                    'contexts': [],
                    'ground_truth': []
                }

                for item in dataset:
                    data_samples['question'].append(item['Frage'])
                    data_samples['answer'].append(item['Answer'])
                    data_samples['contexts'].append(item['Context'].split('The Document is'))  # Kontext in eine Liste aufteilen
                    data_samples['ground_truth'].append(item['Ground_truth'])

                #print("Umgewandelte Datenstruktur:")
                #print(json.dumps(data_samples, indent=4))  # Formatiertes JSON ausgeben
                return data_samples

            except json.JSONDecodeError as e:
                print(f"Fehler beim Laden der JSON-Datei: {e}")
            except Exception as e:
                print(f"Ein Fehler ist aufgetreten: {e}")

    data = loadJSON()

    print(data)
    dataset = Dataset.from_dict(data)

    #generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    print (faithfulness)
    embedding_model = HuggingFaceEmbeddings(
        model_name="H:\\Uni\\Master\\Masterarbeit\\Masterarbeit\\Models\\multilingual-e5-large-instruct")
    os.environ["OPENAI_API_KEY"] = api_key
    eval_model = ChatOpenAI(
        api_key=api_key,
        #base_url=base_url,
        #model = model,
        model="gpt-4o",
        #embeddings=embedding_model
    )
    # Start OpenAI client
    #openai.api_base = base_url
    openai.api_base = "https://api.openai.com/v1/"
    conf = RunConfig(max_retries = 30, max_wait = 300, max_workers=2, timeout=300)
    #metrics = [faithfulness, answer_relevancy, context_precision,context_recall]
    #init_ragas_metrics(metrics, llm=eval_model, embedding=embedding_model)
    #score = evaluate(dataset, metrics=[faithfulness, answer_correctness, ],llm=eval_model, run_config=conf)
    try:
        score = evaluate(dataset, llm=eval_model, run_config=conf, embeddings=embedding_model)
    except Exception as e:
        print(f"Exception:{e}")
    df = score.to_pandas()
    df.to_csv('H:\\Uni\\Master\\Masterarbeit\\score_fin.csv', index=False)
    #df = pandas.read_csv('H:\\Uni\\Master\\Masterarbeit\\score.csv')
    print(df.columns.values)

    #Issue für Faithfulness:
    #https: // github.com / explodinggradients / ragas / issues / 733
    for index,metric in df.iterrows():
        print(metric['user_input'])
        print(metric['answer_relevancy'])
        print(metric['context_precision'])
        print(metric['faithfulness'])
        print(metric['context_recall'])




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
