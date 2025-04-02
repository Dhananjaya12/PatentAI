from rag_implementation import RAG_Retriever
from transformers import pipeline
from llm_response import LLMResponseGenerator
from llm_config import llm_prompts

class PatentSimilarityChecker:
    def __init__(self, retriever: RAG_Retriever, tokenizer, model):
        print("Initializing Patent Similarity Checker...")
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer

        self.llm_response_generator = LLMResponseGenerator(self.model,self.tokenizer)

    def find_similar_patents(self, full_text, article_type="d"):
        formatted_user_content = llm_prompts["similarity-summarizer"]["user_content"].format(full_text=full_text)
        prompt_in_chat_format = [
                                    {"role": "system", "content": llm_prompts["similarity-summarizer"]["preprompt"]},
                                    {"role": "user", "content": formatted_user_content}
                                ]

        summary = self.llm_response_generator.get_prompt_output(prompt_in_chat_format)

        similar_results = self.retriever.get_query_answer(str(article_type), str(summary), "Similar Patents")

        return similar_results
