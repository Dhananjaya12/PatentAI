### Chnagede 
import os
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.spatial.distance import euclidean
# from scipy.spatial.distance import cityblock
from rank_bm25 import BM25Okapi
from loading_resources import Loader
from preprocessing_data import Preprocessor
from llm_response import LLMResponseGenerator
from llm_config import llm_prompts
import time

class RAG_Retriever():
    def __init__(self, gen_model_name, tokenizer, model):
        load_dotenv()
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.gen_model_name = gen_model_name #"microsoft/phi-4" #gpt2, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3.1-8B-Instruct, microsoft/phi-4, meta-llama/Meta-Llama-3.1-8B-Instruct
        self.embedding_model_name = "llama3.1:8b" #"llama3.1:8b", sentence-transformers/all-MiniLM-L6-v2, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3.1-8B-Instruct
        self.index_file_path = 'vector_index.pkl'
        self.top_k = 3
        self.tokenizer = tokenizer
        self.model = model
        self.embed_model = OllamaEmbedding(model_name=self.embedding_model_name)

        Settings.embed_model = self.embed_model
        Settings._tokenizer = self.tokenizer.encode
        Settings.llm = None
        Settings.chunk_size = 2000
        Settings.chunk_overlap = 50

        self.loader = Loader()
        self.preprocessor = Preprocessor(self.tokenizer, self.gen_model_name, self.index_file_path)
        self.llm_response_generator = LLMResponseGenerator(self.model,self.tokenizer)
            
    def prepare_context_response(self, response):
        context = "Context:\n" + "\n\n".join([node.text for node in response.source_nodes[:self.top_k]])
        return context.replace("\n", " ").replace(" ,", ",")
    
    def clean_output(self, generated_text, prompt_text):
        answer_start = generated_text.find("Answer:")
        if answer_start != -1:
            return generated_text[answer_start + len("Answer:"):].strip()
        return generated_text.strip() 
    
    def get_query_answer(self, article_type, user_query, task):
        patent_data = self.loader.load_data(article_type)
        documents = self.preprocessor.create_documents(patent_data)
        print('Number of documents :', len(documents))

        index, tokenized_docs = self.preprocessor.prepare_index(documents, article_type[0])

        retriever = CustomRetriever(index=index, tokenized_docs = tokenized_docs, tokenizer=self.tokenizer, similarity_top_k=self.top_k)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
        )

        start_time = time.time()
        response = query_engine.query(user_query)
        end_time = time.time()
        print(f"Retrieved relevant documents in time: {end_time - start_time:.4f} seconds")

        context = self.prepare_context_response(response)

        formatted_user_content = llm_prompts[task]["user_content"].format(user_query=user_query,context=context)
        prompt_in_chat_format = [
                                    {"role": "system", "content": llm_prompts[task]["preprompt"]},
                                    {"role": "user", "content": formatted_user_content}
                                ]
        
        response = self.llm_response_generator.get_prompt_output(prompt_in_chat_format)
        # print('')
        return response

# Custom retriever class
class CustomRetriever(VectorIndexRetriever):
    def __init__(self, index, tokenized_docs, tokenizer, similarity_top_k=3):
        super().__init__(index=index)  # Initialize the parent class
        self.similarity_top_k = similarity_top_k
        self._index = index  # Store the index in a private attribute
        self.tokenizer = tokenizer
        self.tokenized_docs = tokenized_docs
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
        # tokenized_docs = [self.tokenizer(doc.text)['input_ids'] for doc in documents]  # Tokenize documents 

        bm25 = BM25Okapi(self.tokenized_docs)
        tokenized_query = self.tokenizer(query_bundle.query_str)['input_ids']
        print(f"Query Tokens ({query_bundle.query_str}): {len(tokenized_query)}")
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

        retrieved_text = ' '.join([documents[i].text for i in top_k_indices])
        # print('retrieved text', retrieved_text)
        retrieved_tokens = len(self.tokenizer(retrieved_text)['input_ids'])

        # print(query_bundle.query_str)
        print('Retrieved token length = ',len(self.tokenizer(' '.join([list(self._index.docstore.docs.values())[i].text for i in top_k_indices]))['input_ids'] ))

        return [
            NodeWithScore(node=TextNode(id_=list(self._index.docstore.docs.keys())[i],  # Get the document ID
                                        text=list(self._index.docstore.docs.values())[i].text),
                        score=bm25_scores[i]) 
            for i in top_k_indices
        ]