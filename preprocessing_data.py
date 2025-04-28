from llama_index.core import Document
from llama_index.core import VectorStoreIndex
import os
import pickle
import time

class Preprocessor():
    def __init__(self, tokenizer,gen_model_name):
        self.tokenizer = tokenizer
        self.gen_model_name = gen_model_name

    def create_documents(self, df):
        return [Document(text=f"Abstract: {row['abstract']}\n\nDescription: {row['description']}") for _, row in df.iterrows()]


    def analyze_token_lengths(self, documents, article_type):
        """Analyze the token lengths of the documents and print the distribution."""
        token_lengths = [len(self.tokenizer.encode(doc.text)) for doc in documents]
        # Plot the distribution of token lengths
        plt.figure()  
        plt.hist(token_lengths, bins=20, edgecolor='black')
        plt.title('Token Length Distribution of Documents')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Frequency')
        plot_filename = f'token_length_distribution_{article_type}.png'
        plt.savefig(plot_filename)  # Save the plot as a PNG file
        print(f"Plot saved as {plot_filename}")
        
        return token_lengths

    def prepare_index(self, documents, article_type):

        if os.path.exists(os.path.join('saved_indexes',f'{self.gen_model_name.split("/")[-1]}',f'{self.gen_model_name.split("/")[-1]}_' + 'vector_index' + f'_{article_type}' + '.pkl')):
            start_time = time.time()
            with open(os.path.join('saved_indexes',f'{self.gen_model_name.split("/")[-1]}',f'{self.gen_model_name.split("/")[-1]}_' + 'vector_index' + f'_{article_type}' + '.pkl'), "rb") as f:
                data = pickle.load(f)
            index = data["index"]
            tokenized_docs = data["tokenized_docs"]
            # print('tokenized_docs', tokenized_docs)
            end_time = time.time()
            print(f"Loaded index in time: {end_time - start_time:.4f} seconds")
            print('Number of chunks after indexing :', len(index.vector_store.data.embedding_dict.values()))
        else:
            start_time_index = time.time()
            index = VectorStoreIndex.from_documents(documents)
            text_chunks = list(index.docstore.docs.values())
            start_time_tokenizing = time.time()
            tokenized_docs = [self.tokenizer(chunk.text)['input_ids'] for chunk in text_chunks]
            end_time_tokenizing = time.time()
            print(f"Tokenized docs in time: {end_time_tokenizing - start_time_tokenizing:.4f} seconds")
            
            with open(os.path.join('saved_indexes',f'{self.gen_model_name.split("/")[-1]}',f'{self.gen_model_name.split("/")[-1]}_' + 'vector_index' + f'_{article_type}' + '.pkl'), "wb") as f:
                pickle.dump({"index": index, "tokenized_docs": tokenized_docs}, f)

            end_time_index = time.time()
            print(f"Created new indexes in time: {end_time_index - start_time_index:.4f} seconds")
            print('Number of chunks after indexing :', len(index.vector_store.data.embedding_dict.values()))
        return index, tokenized_docs