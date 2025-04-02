llm_prompts = {"RAG":{"preprompt":"""You are a helpful assistant that answers questions using only the provided context.
                                        Output **only the answer**, and nothing else.
                                        Do not include any explanations, extra text, or formatting.
                                        If the answer cannot be found in the context, respond only with: 'Insufficient Information'.
                                    """,

                        "user_content":"""Context:\n{context}\n
                                            Question: {user_query}\n
                                            Answer:"""

                    },

    "Similar Patents":{  "preprompt":"""
                                        You are a domain expert in patent law and intellectual property analysis. Your job is to determine whether a new patent idea is already disclosed in any previously filed patents.

                                        Focus on conceptual similarities even if the wording differs. Be concise, critical, and use logical reasoning.

                                        Respond only in one of these formats:
                                        - "Novel: [brief reasoning]"
                                        - "Not Novel: Overlaps with Patent [ID or Title] - [brief reasoning]"
                                        """,
                       
                        "user_content": """Below is a new patent summary followed by context from previously 
                                            filed similar patents. Your task is to determine whether the new patent presents novel 
                                            ideas or overlaps with existing disclosures.
                                            New Patent: {user_query},
                                            Retrieved Prior Art: {context}
                                            Question: Is the new patent novel, or does it overlap with any of the prior documents? Justify your answer briefly.
                                        """
                                        
                    },

    "similarity-summarizer":{"preprompt":"""You are a helpful assistant that summarizes patent documents clearly and concisely. 
                    Focus on the core technical ideas, innovations, and unique features. 
                    Preserve all key details while avoiding repetition or irrelevant background.
                """,
                "user_content":"""Summarize the following patent document by extracting its essential and most important technical points:\n\n{full_text}\n\nSummary:"""
                },

    "extracter":{"preprompt":"""You are tasked with extracting specific sections from the patent text. Do not return any context, input text, or explanations.\n\n
                Extract only the following two sections from the patent text:\n\n
                1. Claims: These are the formal claims made in the patent. List all the claims as clearly as possible.\n
                2. Figures: These are the descriptions of the figures or drawings included in the patent, if any exist.\n\n
                You must return your response exactly in this format. Do not add anything else, and do not repeat any part of the input text:\n
                ***Claims***:\n<claims_text_here>\n\n***Figures***:\n<figure_descriptions_here>\n\n
                Do not include any extra details, summaries, or restatements of the original patent text. Only provide the extracted content. No further explanation is required.\n\n
                """,
                "user_content":"""The input Patent document is
                                    {text}"""
                },

    "extracter-summarizer":{"preprompt":"""You are a helpful assistant that summarizes patent sections clearly and concisely. 
                                            Preserve all key technical details, but avoid unnecessary repetition or filler.
                                        """,
                                        
                            "user_content":"""Summarize the following patent section:\n\n{text}\n\nSummary:"""}                                    
                                        
                                        
            }