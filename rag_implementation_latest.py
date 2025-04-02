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
import matplotlib.pyplot as plt

class RAG_Retriever():
    def __init__(self):
        print('loaded requirements')
        load_dotenv()
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.gen_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" #gpt2, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3.1-8B-Instruct
        self.embedding_model_name = "llama3.1:8b" #"llama3.1:8b", sentence-transformers/all-MiniLM-L6-v2, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3.1-8B-Instruct
        self.index_file_path = 'vector_index.pkl'
        self.top_k = 2

        self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name, token=self.access_token, legacy = False)
        self.model = AutoModelForCausalLM.from_pretrained(self.gen_model_name, token=self.access_token)
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set embedding model
        self.embed_model = OllamaEmbedding(model_name=self.embedding_model_name)

        Settings.embed_model = self.embed_model
        Settings._tokenizer = self.tokenizer.encode
        Settings.llm = None
        Settings.chunk_size = 5300
        Settings.chunk_overlap = 25

    def load_data(self,codes):
        patent_data = load_dataset("big_patent",codes=codes,trust_remote_code=True, split='test', streaming=True) 
        data = list(patent_data)  
        df = pd.DataFrame(data)
        return df
    
    def create_documents(self, df):
        return [Document(text=row['description']) for _, row in df.iterrows()]
    
    def analyze_token_lengths(self, documents):
        """Analyze the token lengths of the documents and print the distribution."""
        token_lengths = [len(self.tokenizer.encode(doc.text)) for doc in documents]
        # Plot the distribution of token lengths
        plt.hist(token_lengths, bins=20, edgecolor='black')
        plt.title('Token Length Distribution of Documents')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Frequency')
        plot_filename = 'token_length_distribution.png'
        plt.savefig(plot_filename)  # Save the plot as a PNG file
        print(f"Plot saved as {plot_filename}")
        
        return token_lengths
    
    def prepare_index(self,documents):
        if os.path.exists(self.index_file_path):
            with open(self.index_file_path, "rb") as f:
                index = pickle.load(f)
            print("Loaded existing index from the file.")
            print('Number of chunks after indexing :', len(index.vector_store.data.embedding_dict.values()))
        else:
            index = VectorStoreIndex.from_documents(documents)
            with open(self.index_file_path, "wb") as f:
                pickle.dump(index, f)
            print("Created a new index and saved it to the file.")
            print('Number of chunks after indexing :', len(index.vector_store.data.embedding_dict.values()))
        return index

    # def analysis_function(row):
    #     texts = [doc.text for doc in self.index.docstore.docs.values()]
    #     for similarity, text in zip(row['similarity'], texts):
    #         if row['fact'] in text:
    #             return similarity
            
    def prepare_context_response(self, response):
        context = "Context:\n" + "\n\n".join([node.text for node in response.source_nodes[:self.top_k]])
        return context.replace("\n", " ").replace(" ,", ",")
    
    def clean_output(self, generated_text, prompt_text):
            response_start = generated_text.find(prompt_text)
            cleaned_text = generated_text[response_start + len(prompt_text):].strip() if response_start != -1 else generated_text.strip()
            
            # Check if ':' exists in the cleaned text
            colon_index = cleaned_text.find(':')
            if colon_index != -1:
                return cleaned_text[colon_index + 1:].strip()  # Take everything after ':'
            
            return cleaned_text
    
    def get_prompt_output(self, prompt):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            output_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                pad_token_id= self.tokenizer.pad_token_id
            )
            output = self.clean_output(self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip(), prompt)
            # output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            return output

    
    def get_query_answer(self, article_type, user_query):
        patent_data = self.load_data(codes=article_type)
        print('length of patent_data', len(patent_data))

        documents = self.create_documents(patent_data[:10])
        print('Number of documents :', len(documents))

        document_token_list = self.analyze_token_lengths(documents)

        index = self.prepare_index(documents)

        retriever = CustomRetriever(index=index, tokenizer=self.tokenizer, similarity_top_k=self.top_k)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
        )

        response = query_engine.query(user_query)

        context = self.prepare_context_response(response)

        prompt_with_context = (f"Below is a question followed by some context from different sources." 
                            f"Please answer the question based on the context. The answer to the question is a word or entity." 
                            f"If the provided information is insufficient to answer the question, respond 'Insufficient Information'." 
                            f"Answer directly without explanation."
                            f"Context: {context}\nQuestion: {user_query}")
        
        print('prompt_with_context', prompt_with_context)
        
        return self.get_prompt_output(prompt_with_context)

# Custom retriever class
class CustomRetriever(VectorIndexRetriever):
    def __init__(self, index, tokenizer, similarity_top_k=3):
        super().__init__(index=index)  # Initialize the parent class
        self.similarity_top_k = similarity_top_k
        self._index = index  # Store the index in a private attribute
        self.tokenizer = tokenizer
        self.similarity_list_dict = {'cosine':[],'euclidean':[],'manhatten':[],'bm25':[]}

    def get_query_embedding(self, query_str):
        """Get the embedding for the query string."""
        return Settings.embed_model.get_query_embedding(query_str)  # Assuming embed returns a list

    def retrieve(self, query_bundle):
        """Retrieve the top-k documents based on cosine similarity, Euclidean distance, Manhattan distance, and BM25 score."""

        # query_vector = self.get_query_embedding(query_bundle.query_str)
        # cosine_similarities = []
        # euclidean_similarities = []
        # manhatten_similarities = []
        bm25_scores = []

        # Retrieve vectors from the index
        # doc_vectors = list(self._index.vector_store.data.embedding_dict.values())
        documents = list(self._index.docstore.docs.values())
        
        # Calculate BM25 scores
        tokenized_docs = [self.tokenizer(doc.text)['input_ids'] for doc in documents]  # Tokenize documents
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = self.tokenizer(query_bundle.query_str)['input_ids']
        # tokenized_query = query_bundle.query_str.split()  # Tokenize the query
        bm25_scores = bm25.get_scores(tokenized_query)

        # for doc_vector in doc_vectors:  # Use self._index
        #     cosine_sim = cosine_similarity(np.array(query_vector).reshape(1, -1), np.array(doc_vector).reshape(1, -1))[0][0]
        #     cosine_similarities.append(cosine_sim)
        #     euclidean_sim = 1 / (1 + euclidean(np.array(query_vector), np.array(doc_vector)))
        #     euclidean_similarities.append(euclidean_sim)
        #     manhatten_sim = 1 / (1 + cityblock(np.array(query_vector), np.array(doc_vector)))
        #     manhatten_similarities.append(manhatten_sim)

        # self.similarity_list_dict['cosine'].append(cosine_similarities)
        # self.similarity_list_dict['euclidean'].append(euclidean_similarities)
        # self.similarity_list_dict['manhatten'].append(manhatten_similarities)
        # self.similarity_list_dict['bm25'].append(bm25_scores)  # Store BM25 scores

        # Get indices of top-k similar documents based on Manhattan similarities
        top_k_indices = np.argsort(bm25_scores)[-self.similarity_top_k:][::-1]

        print(query_bundle.query_str)
        print('token length = ',len(self.tokenizer(' '.join([list(self._index.docstore.docs.values())[i].text for i in top_k_indices]))['input_ids'] ))
        
        return [
            NodeWithScore(node=TextNode(id_=list(self._index.docstore.docs.keys())[i],  # Get the document ID
                                        text=list(self._index.docstore.docs.values())[i].text),
                        score=bm25_scores[i]) 
            for i in top_k_indices
        ]