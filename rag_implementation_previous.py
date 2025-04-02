import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from transformers import AutoTokenizer, AutoModelForCausalLM # Use for llama3 models
from rank_bm25 import BM25Okapi
from datasets import load_dataset
# from transformers import LlamaTokenizer, LlamaForCausalLM
# from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Load environment variables
print('loaded requirements')
load_dotenv()
access_token = os.getenv("ACCESS_TOKEN")

# Configurable settings
# use_sample_data = False
# prepare_eval_set = True
gen_model_name = "microsoft/phi-4" #gpt2, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3.1-8B-Instruct
embedding_model_name = "llama3.1:8b" #"llama3.1:8b", sentence-transformers/all-MiniLM-L6-v2, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3.1-8B-Instruct
index_file_path = 'vector_index.pkl'
top_k = 2

# Load LLaMA tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(gen_model_name, token=access_token, legacy = False)
model = AutoModelForCausalLM.from_pretrained(gen_model_name, token=access_token)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.pad_token = tokenizer.eos_token

# Set embedding model
embed_model = OllamaEmbedding(model_name=embedding_model_name)

Settings.embed_model = embed_model
Settings._tokenizer = tokenizer.encode
Settings.llm = None
Settings.chunk_size = 5300
Settings.chunk_overlap = 25

# Load sample data or full corpus
# def load_data():
#     if use_sample_data:
#         filter_title = [
#                         "Travis Kelce says he tried to delete his 'nonsense' tweets before they went viral",
#                         "Manchester United face Galatasaray with high hopes but bad memories",
#                         "Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver",
#                         "Martin Scorsese gets quizzed by his daughter on internet slang"
#                                 ]
        
#         with open('/raid/huyen/qa_eval/paliwal/RAG_Framework_Implementation/datasource/corpus.json') as file:
#             corpus_data = pd.json_normalize(json.load(file))
#             corpus_data = corpus_data[corpus_data['title'].isin(filter_title)]
#             print(corpus_data)
#         with open('/raid/huyen/qa_eval/paliwal/RAG_Framework_Implementation/datasource/MultiHopRAG.json') as file:
#             multihog_data = pd.json_normalize(
#                 json.load(file),
#                 record_path=['evidence_list'],
#                 meta=['query', 'answer', 'question_type'],
#                 meta_prefix='',
#                 errors='ignore')
#             multihog_data = multihog_data[multihog_data['title'].isin(filter_title)]
#     else:
#         with open('/raid/huyen/qa_eval/paliwal/RAG_Framework_Implementation/datasource/corpus.json') as file:
#             corpus_data = pd.json_normalize(json.load(file))
        
#         with open('/raid/huyen/qa_eval/paliwal/RAG_Framework_Implementation/datasource/MultiHopRAG.json') as file:
#             multihog_data = pd.json_normalize(
#                 json.load(file),
#                 record_path=['evidence_list'],
#                 meta=['query', 'answer', 'question_type'],
#                 meta_prefix='',
#                 errors='ignore'
#             )
    
#     return corpus_data, multihog_data

def load_data():
    patent_data = load_dataset("big_patent", "d",trust_remote_code=True, split='test', streaming=True) 
    data = list(patent_data)  
    df = pd.DataFrame(data)
    return df

patent_data = load_data()
print('length of patent_data', len(patent_data))


# if prepare_eval_set:
#     grouped_data = multihog_data.groupby('query')

#     # Step 2: Randomly select 3 queries
#     random_queries = random.sample(list(grouped_data.groups.keys()), 3)

#     # Initialize a dictionary to store all articles for each query group
#     query_article_dict = {}

#     # Initialize a list to store all articles (relevant and irrelevant combined)
#     all_selected_articles = []

#     # Step 3: For each selected query, get all related articles and 3 irrelevant articles
#     for query in random_queries:
#         # Get relevant articles for the current query
#         relevant_articles = grouped_data.get_group(query)['title'].tolist()
        
#         # Get irrelevant articles (articles that don't belong to the current query)
#         remaining_articles = multihog_data[~(multihog_data['query'].isin([query]) | multihog_data['title'].isin([relevant_articles]))]
#         irrelevant_articles = remaining_articles.sample(n=3)['title'].tolist()
        
#         # Combine relevant and irrelevant articles into one list
#         all_articles = relevant_articles + irrelevant_articles
        
#         # Store all articles in the dictionary for later use
#         query_article_dict[query] = all_articles

#         # Add the articles to the overall list
#         all_selected_articles.extend(all_articles)

#     # Step 4: Filter the corpus_data using the combined list of all selected articles
#     corpus_data = corpus_data[corpus_data['title'].isin(all_selected_articles)]
#     multihog_data = multihog_data[multihog_data['query'].isin(random_queries)]
    

# Convert DataFrame to a list of Document objects
def create_documents(df):
    return [Document(text=row['description']) for _, row in df.iterrows()]

documents = create_documents(patent_data[:10])
print('Number of documents :', len(documents))

# index = VectorStoreIndex.from_documents(documents)
# print('Number of chunks after indexing :', len(index.vector_store.data.embedding_dict.values()))
# with open(index_file_path, "wb") as f:
#     pickle.dump(index, f)


# Load or create vector store index
if os.path.exists(index_file_path):
    with open(index_file_path, "rb") as f:
        index = pickle.load(f)
    print("Loaded existing index from the file.")
    print('Number of chunks after indexing :', len(index.vector_store.data.embedding_dict.values()))
else:
    index = VectorStoreIndex.from_documents(documents)
    with open(index_file_path, "wb") as f:
        pickle.dump(index, f)
    print("Created a new index and saved it to the file.")
    print('Number of chunks after indexing :', len(index.vector_store.data.embedding_dict.values()))


# Custom retriever class
class CustomRetriever(VectorIndexRetriever):
    def __init__(self, index, similarity_top_k=3):
        super().__init__(index=index)  # Initialize the parent class
        self.similarity_top_k = similarity_top_k
        self._index = index  # Store the index in a private attribute

    def get_query_embedding(self, query_str):
        """Get the embedding for the query string."""
        return Settings.embed_model.get_query_embedding(query_str)  # Assuming embed returns a list

    def retrieve(self, query_bundle):
        """Retrieve the top-k documents based on cosine similarity, Euclidean distance, Manhattan distance, and BM25 score."""
        global similarity_list_dict

        query_vector = self.get_query_embedding(query_bundle.query_str)
        cosine_similarities = []
        euclidean_similarities = []
        manhatten_similarities = []
        bm25_scores = []

        # Retrieve vectors from the index
        doc_vectors = list(self._index.vector_store.data.embedding_dict.values())
        documents = list(self._index.docstore.docs.values())
        
        # Calculate BM25 scores
        tokenized_docs = [tokenizer(doc.text)['input_ids'] for doc in documents]  # Tokenize documents
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = tokenizer(query_bundle.query_str)['input_ids']
        # tokenized_query = query_bundle.query_str.split()  # Tokenize the query
        bm25_scores = bm25.get_scores(tokenized_query)

        for doc_vector in doc_vectors:  # Use self._index
            cosine_sim = cosine_similarity(np.array(query_vector).reshape(1, -1), np.array(doc_vector).reshape(1, -1))[0][0]
            cosine_similarities.append(cosine_sim)
            euclidean_sim = 1 / (1 + euclidean(np.array(query_vector), np.array(doc_vector)))
            euclidean_similarities.append(euclidean_sim)
            manhatten_sim = 1 / (1 + cityblock(np.array(query_vector), np.array(doc_vector)))
            manhatten_similarities.append(manhatten_sim)

        similarity_list_dict['cosine'].append(cosine_similarities)
        similarity_list_dict['euclidean'].append(euclidean_similarities)
        similarity_list_dict['manhatten'].append(manhatten_similarities)
        similarity_list_dict['bm25'].append(bm25_scores)  # Store BM25 scores

        # Get indices of top-k similar documents based on Manhattan similarities
        top_k_indices = np.argsort(bm25_scores)[-self.similarity_top_k:][::-1]

        print(query_bundle.query_str)
        print('token length = ',len(tokenizer(' '.join([list(self._index.docstore.docs.values())[i].text for i in top_k_indices]))['input_ids'] ))
        
        return [
            NodeWithScore(node=TextNode(id_=list(self._index.docstore.docs.keys())[i],  # Get the document ID
                                        text=list(self._index.docstore.docs.values())[i].text),
                        score=bm25_scores[i]) 
            for i in top_k_indices
        ]

    
def analysis_function(row):
    # Extract document texts
    texts = [doc.text for doc in index.docstore.docs.values()]

    # Check if any fact is in the texts and return the corresponding similarity
    for similarity, text in zip(row['similarity'], texts):
        if row['fact'] in text:
            return similarity

# Initialize retriever and query engine
retriever = CustomRetriever(index=index, similarity_top_k=top_k)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
)

# Query the MultiHopRAG data and extract context
similarity_list_dict = {'cosine':[],'euclidean':[],'manhatten':[],'bm25':[]}
patent_data['response'] = patent_data.apply(lambda row: query_engine.query(row['query']), axis=1)
patent_data['cosine_similarity'] = [np.round(lst, 5).tolist() for lst in similarity_list_dict['cosine']]
patent_data['similarity'] = patent_data['cosine_similarity']
patent_data['cosine_fact_similarity'] = patent_data.apply(analysis_function, axis=1).round(5)
patent_data['cosine_sorted_similarity'] = patent_data['cosine_similarity'].apply(lambda x: sorted(x, reverse=True))
patent_data['cosine_fact_score_position'] = patent_data.apply(lambda row: row['cosine_sorted_similarity'].index(row['cosine_fact_similarity']) if row['cosine_fact_similarity'] in row['cosine_sorted_similarity'] else None, axis=1)

patent_data['euclidean_similarity'] = [np.round(lst, 5).tolist() for lst in similarity_list_dict['euclidean']]
patent_data['similarity'] = patent_data['euclidean_similarity'] 
patent_data['euclidean_fact_similarity'] = patent_data.apply(analysis_function, axis=1).round(5)
patent_data['euclidean_sorted_similarity'] = patent_data['euclidean_similarity'].apply(lambda x: sorted(x, reverse=True))
patent_data['euclidean_fact_score_position'] = patent_data.apply(lambda row: row['euclidean_sorted_similarity'].index(row['euclidean_fact_similarity']) if row['euclidean_fact_similarity'] in row['euclidean_sorted_similarity'] else None, axis=1)

patent_data['manhatten_similarity'] = [np.round(lst, 5).tolist() for lst in similarity_list_dict['manhatten']]
patent_data['similarity'] = patent_data['manhatten_similarity']
patent_data['manhatten_fact_similarity'] = patent_data.apply(analysis_function, axis=1).round(5)
patent_data['manhatten_sorted_similarity'] = patent_data['manhatten_similarity'].apply(lambda x: sorted(x, reverse=True))
patent_data['manhatten_fact_score_position'] = patent_data.apply(lambda row: row['manhatten_sorted_similarity'].index(row['manhatten_fact_similarity']) if row['manhatten_fact_similarity'] in row['manhatten_sorted_similarity'] else None, axis=1)

patent_data['bm25_similarity'] = [np.round(lst, 5).tolist() for lst in similarity_list_dict['bm25']]
patent_data['similarity'] = patent_data['bm25_similarity']
patent_data['bm25_fact_similarity'] = patent_data.apply(analysis_function, axis=1).round(5)
patent_data['bm25_sorted_similarity'] = patent_data['bm25_similarity'].apply(lambda x: sorted(x, reverse=True))
patent_data['bm25_fact_score_position'] = patent_data.apply(lambda row: row['bm25_sorted_similarity'].index(row['bm25_fact_similarity']) if row['bm25_fact_similarity'] in row['bm25_sorted_similarity'] else None, axis=1)


# multihog_data['response'] = [query_engine.query(question) for question in multihog_data['query']]

def context_response(response):
    context = "Context:\n" + "\n\n".join([node.text for node in response.source_nodes[:top_k]])
    return context.replace("\n", " ").replace(" ,", ",")

patent_data['context'] = patent_data['response'].apply(context_response)

# multihog_data.to_excel(r'multihog_analysis.xlsx')
# print(ajiwjiw)
# Function to generate answers using the model
def get_query_answer(context, query):
    
    def clean_output(generated_text, prompt_text):
        response_start = generated_text.find(prompt_text)
        cleaned_text = generated_text[response_start + len(prompt_text):].strip() if response_start != -1 else generated_text.strip()
        
        # Check if ':' exists in the cleaned text
        colon_index = cleaned_text.find(':')
        if colon_index != -1:
            return cleaned_text[colon_index + 1:].strip()  # Take everything after ':'
        
        return cleaned_text

    def get_prompt_output(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id
        )
        output = clean_output(tokenizer.decode(output_ids[0], skip_special_tokens=True).strip(), prompt)
        return output

    prompt_with_context = (f"Below is a question followed by some context from different sources." 
                           f"Please answer the question based on the context. The answer to the question is a word or entity." 
                           f"If the provided information is insufficient to answer the question, respond 'Insufficient Information'." 
                           f"Answer directly without explanation."
                           f"Context: {context}\nQuestion: {query}")
    return get_prompt_output(prompt_with_context)

# # Generate answers for each query
#multihog_data['context'] = multihog_data['fact']
patent_data['rag_answer'] = patent_data.apply(lambda row: get_query_answer(row['context'], row['query']), axis=1)
# # Save output to Excel
patent_data.to_excel('/home/paliwal/RAG_Framework_Implementation/outputs/rag_output_answers.xlsx', index=False)
