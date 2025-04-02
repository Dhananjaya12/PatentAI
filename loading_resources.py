import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Use for llama3 models
import torch
from datasets import load_dataset
import time

class Loader():
    def load_models():
        access_token = os.getenv("ACCESS_TOKEN")
        gen_model_name =  "microsoft/phi-4" #gpt2, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3.1-8B-Instruct, microsoft/phi-4, meta-llama/Meta-Llama-3.1-8B-Instruct
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(gen_model_name,token=access_token, truncation=True)
        end_time = time.time()
        print(f"Loaded Tokenizer in time: {end_time - start_time:.4f} seconds")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(gen_model_name, quantization_config=bnb_config, token=access_token)
        end_time = time.time()
        model.to(device)
        print(f"Loaded model in time: {end_time - start_time:.4f} seconds")
        return gen_model_name, tokenizer, model
    
    def load_data(self, codes):
        start_time = time.time()
        patent_data = load_dataset("big_patent", codes=codes,trust_remote_code=True, split='test', streaming=True) 
        data = list(patent_data)  
        df = pd.DataFrame(data)
        end_time = time.time()
        print(f"Loaded dataset in time: {end_time - start_time:.4f} seconds")
        print("Length of patent_data", len(df))
        return df