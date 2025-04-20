
import re
import pandas as pd
from dotenv import load_dotenv
from preprocessing_data import Preprocessor
from loading_resources import Loader
import torch
from llm_response import LLMResponseGenerator
from llm_config import llm_prompts

class PatentExtractor:
    def __init__(self, gen_model_name, tokenizer, model):
        self.gen_model_name = gen_model_name
        self.tokenizer = tokenizer
        self.model = model

        self.loader = Loader()
        self.llm_response_generator = LLMResponseGenerator(self.model,self.tokenizer)
        self.preprocessor = Preprocessor(self.tokenizer, self.gen_model_name)

    def extract_sections(self, generated):
        # Define regex patterns for claims and figures
        claims_pattern = r"\*\*\*Claims\*\*\*:\s*(.*?)(?=\*\*\*Figure Descriptions\*\*\*|$)"
        figures_pattern = r"\*\*\*Figure Descriptions\*\*\*:\s*(.*)"

        claims_match = re.search(claims_pattern, generated, re.DOTALL)
        claims = claims_match.group(1).strip() if claims_match else ""

        figures_match = re.search(figures_pattern, generated, re.DOTALL)
        figures = figures_match.group(1).strip() if figures_match else ""
        return claims, figures
    
    def process_patents(self, article_type, index):
        """Extract and summarize key sections from patent documents."""
        patent_data = self.loader.load_data(article_type)
        documents = self.preprocessor.create_documents(patent_data.iloc[[index]])
        print('Number of documents :', len(documents))
        
        results = []
        for doc in documents:
            abstract = doc.text.split("Abstract:")[1].split("Description:")[0].strip()

            formatted_user_content = llm_prompts["extracter"]["user_content"].format(text=doc.text.split("Description:")[1].strip())
            prompt_in_chat_format = [
                                        {"role": "system", "content": llm_prompts["extracter"]["preprompt"]},
                                        {"role": "user", "content": formatted_user_content}
                                    ]

            sections = self.llm_response_generator.get_prompt_output(prompt_in_chat_format)
            claims, drawings = self.extract_sections(sections)
            
            formatted_user_content = llm_prompts["extracter-summarizer"]["user_content"].format(text=claims)

            prompt_in_chat_format = [
                                        {"role": "system", "content": llm_prompts["extracter-summarizer"]["preprompt"]},
                                        {"role": "user", "content": formatted_user_content}
                                    ]

            claims_summary = self.llm_response_generator.get_prompt_output(prompt_in_chat_format)

            formatted_user_content = llm_prompts["extracter-summarizer"]["user_content"].format(text=drawings)

            prompt_in_chat_format = [
                                        {"role": "system", "content": llm_prompts["extracter-summarizer"]["preprompt"]},
                                        {"role": "user", "content": formatted_user_content}
                                    ]

            drawings_summary = self.llm_response_generator.get_prompt_output(prompt_in_chat_format)
            results.append(f"Abstract: {abstract}\nClaims Summary: {claims_summary}\nFigures Summary: {drawings_summary}\n")
        return results
