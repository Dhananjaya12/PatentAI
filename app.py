# from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QComboBox
# import sys
# from rag_implementation import RAG_Retriever

# class RAGInterface(QWidget):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("RAG Interface")
#         self.setGeometry(100, 100, 500, 400)

#         layout = QVBoxLayout()

#         # Dropdown Label
#         self.label = QLabel("Select Article Type:")
#         layout.addWidget(self.label)

#         # Dropdown (ComboBox)
#         self.article_dropdown = QComboBox()
#         self.article_dropdown.addItems(["all",
#             "Human Necessities", "Performing Operations; Transporting", "Chemistry; Metallurgy",
#             "Textiles; Paper", "Fixed Constructions", "Mechanical Engineering; Lightning; Heating; Weapons; Blasting", 
#             "Physics", "Electricity", "General tagging of new or cross-sectional technology"
#         ])
#         # self.article_dropdown.addItems([
#         #     "a", "b", "c", "d"
#         # ])
#         layout.addWidget(self.article_dropdown)

#         # Query input label
#         self.query_label = QLabel("Enter Query:")
#         layout.addWidget(self.query_label)

#         # Query input box
#         self.query_textbox = QTextEdit()
#         layout.addWidget(self.query_textbox)

#         # Submit button
#         self.submit_button = QPushButton("Get Answer")
#         self.submit_button.clicked.connect(self.process_query)
#         layout.addWidget(self.submit_button)

#         # Answer label
#         self.answer_label = QLabel("Answer:")
#         layout.addWidget(self.answer_label)

#         # Answer text box
#         self.answer_textbox = QTextEdit()
#         self.answer_textbox.setReadOnly(True)
#         layout.addWidget(self.answer_textbox)

#         self.setLayout(layout)

#     def process_query(self):
#         atricle_type_mapping = {"all":"all","Human Necessities":"a", "Performing Operations; Transporting":"b", "Chemistry; Metallurgy":"c",
#             "Textiles; Paper":"d", "Fixed Constructions":"e", "Mechanical Engineering; Lightning; Heating; Weapons; Blasting":"f", 
#             "Physics":"g", "Electricity":"h", "General tagging of new or cross-sectional technology":"y"}
        
#         article_type = atricle_type_mapping[str(self.article_dropdown.currentText())]
#         user_query = self.query_textbox.toPlainText().strip()

#         print('article_type', article_type)
#         print('user_query',user_query)

#         if not user_query:
#             self.answer_textbox.setPlainText("Please enter a query.")
#             return

#         # Call RAG Retriever
#         response = RAG_Retriever().get_query_answer([str(article_type)], str(user_query))

#         # Display response
#         self.answer_textbox.setPlainText(response)

# # Run the Application
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = RAGInterface()
#     window.show()
#     sys.exit(app.exec())


##Code for CLI mode
from rag_implementation import RAG_Retriever
from loading_resources import Loader
from extract_summarize import PatentExtractor
from similarity_checker import PatentSimilarityChecker

def main():
    gen_model_name, tokenizer, model = Loader.load_models()

    while True:
        try:
            task = input("Select the task number to perform: \n1. RAG \n2. Extract and summarize information\n3. Find similar patents\n4. Exit\nEnter choice: ").strip()
            
            if task == "4" or task.lower() == "exit":
                print("Exiting...")
                break
            
            elif task == "1" or task == "2" or task == "3":
                article_types = ["a: Human Necessities", "b: Performing Operations; Transporting", 
                                "c: Chemistry; Metallurgy", "d: Textiles; Paper", "e: Fixed Constructions", 
                                "f: Mechanical Engineering; Lightning; Heating; Weapons; Blasting", "g: Physics", 
                                "h: Electricity", "y: General tagging of new or cross-sectional technology"]
                
                for idx, article in enumerate(article_types, start=1):
                    print(f"{idx}. {article}")

                choice = input("\nSelect an article type (1-9) or type 'exit' to quit: ").strip()
                if choice.lower() == "exit":
                    print("Exiting...")
                    break
                try:
                    choice = int(choice) - 1
                    if choice < 0 or choice >= len(article_types):
                        raise ValueError("Invalid choice")
                    article_type = article_types[choice]
                except ValueError as e:
                    print(f"Invalid selection: {e}. Try again.")
                    continue

                if task == "1":
                    print("Welcome to the RAG Interface (CLI Mode)")
                    user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
                
                    if user_query.lower() == "exit":
                        print("Exiting...")
                        break
                    if not user_query:
                        print("Error: Query cannot be empty.")
                        continue
                    
                    print("\nProcessing your query...\n")
                    try:
                        response = RAG_Retriever(gen_model_name, tokenizer, model,3).get_query_answer([str(article_type)], str(user_query), str("RAG"))
                        print("\nRAG Retriever Output:\n" + response)
                    except Exception as e:
                        print(f"Error while processing query: {e}")

                elif task == "2":
                    document_index = input("\nSelect the document index which you want information for (or type 'exit' to quit): ").strip()
                    if document_index.lower() == "exit":
                        print("Exiting...")
                        break
                    try:
                        document_index = int(document_index)
                    except ValueError:
                        print("Invalid document index. Please enter a valid number.")
                        continue
                
                    try:
                        extractor = PatentExtractor(gen_model_name, tokenizer, model)
                        results = extractor.process_patents(article_type, document_index)
                        print('Extract and Summarize Information Output:\n')
                        for result in results:
                            print(result)
                    except Exception as e:
                        print(f"Error while processing patent extraction: {e}")
                    
                elif task == "3":
                    full_text = input("\nGive the content of the new Patent filing (or type 'exit' to quit): ")
                    if full_text.lower() == "exit":
                        print("Exiting...")
                        break
                    try:
                        full_text = str(full_text)
                    except ValueError:
                        print("Invalid document index. Please enter a valid text.")
                        continue
                
                    try:
                        retriever = RAG_Retriever(gen_model_name, tokenizer, model,1)
                        similarity_finder = PatentSimilarityChecker(retriever, tokenizer, model)
                        results = similarity_finder.find_similar_patents(full_text,article_type)
                        print('Find Similar Patents:\n',results)
                    except Exception as e:
                        print(f"Error while processing patent extraction: {e}")

            else:
                print("Invalid input. Please enter 1, 2, or 3 (exit).")
                continue
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

if __name__ == "__main__":
    main()
