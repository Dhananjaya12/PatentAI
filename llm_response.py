from transformers import pipeline
import time

class LLMResponseGenerator():
    def __init__(self, model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_prompt_output(self, prompt_in_chat_format):
        start_time = time.time()
        llm_pipeline = pipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                task = "text-generation",
                batch_size=4,
                do_sample=True,
                temperature = 0.0000000001,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens= 500,
                truncation=True
        )

        llm_prompt_template = self.tokenizer.apply_chat_template(
                prompt_in_chat_format, tokenize=False, add_generation_prompt=True
            )

        response = llm_pipeline(llm_prompt_template)[0]['generated_text']
        end_time = time.time()
        print(f"LLM response in time: {end_time - start_time:.4f} seconds")
        return response