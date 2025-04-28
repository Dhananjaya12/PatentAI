from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QComboBox, QMessageBox, QHBoxLayout
)
import sys
from rag_implementation import RAG_Retriever
from loading_resources import Loader
from extract_summarize import PatentExtractor
from similarity_checker import PatentSimilarityChecker

class RAGInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patent Analysis Toolkit")
        self.setGeometry(100, 100, 600, 500)

        self.layout = QVBoxLayout()

        self.task_label = QLabel("Select Task:")
        self.layout.addWidget(self.task_label)

        self.task_dropdown = QComboBox()
        self.task_dropdown.addItems(["RAG", "Extract and summarize information", "Find similar patents"])
        self.task_dropdown.currentIndexChanged.connect(self.update_input_fields)
        self.layout.addWidget(self.task_dropdown)

        self.article_label = QLabel("Select Article Type:")
        self.layout.addWidget(self.article_label)

        self.article_dropdown = QComboBox()
        self.article_mapping = {
            "Human Necessities": "a",
            "Performing Operations; Transporting": "b",
            "Chemistry; Metallurgy": "c",
            "Textiles; Paper": "d",
            "Fixed Constructions": "e",
            "Mechanical Engineering; Lightning; Heating; Weapons; Blasting": "f",
            "Physics": "g",
            "Electricity": "h",
            "General tagging of new or cross-sectional technology": "y"
        }
        self.article_dropdown.addItems(self.article_mapping.keys())
        self.layout.addWidget(self.article_dropdown)

        # Dynamic input label
        self.input_label = QLabel()
        self.layout.addWidget(self.input_label)

        # Input widgets
        self.input_textbox = QTextEdit()
        self.input_dropdown = QComboBox()
        self.input_dropdown.hide()
        self.layout.addWidget(self.input_textbox)
        self.layout.addWidget(self.input_dropdown)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_task)
        self.layout.addWidget(self.run_button)

        self.output_label = QLabel("Output:")
        self.layout.addWidget(self.output_label)

        self.output_textbox = QTextEdit()
        self.output_textbox.setReadOnly(True)
        self.layout.addWidget(self.output_textbox)

        self.setLayout(self.layout)

        # Placeholder for model loading
        self.gen_model_name, self.tokenizer, self.model = Loader.load_models()

        # Initialize UI state
        self.update_input_fields()

    def update_input_fields(self):
        task = self.task_dropdown.currentText()

        if task == "RAG":
            self.input_label.setText("Enter your Query:")
            self.input_textbox.show()
            self.input_dropdown.hide()
        elif task == "Extract and summarize information":
            self.input_label.setText("Select Document:")
            self.input_textbox.hide()
            self.input_dropdown.clear()
            list_of_doc_headings = ["Document 1", "Document 2", "Document 3"]  # Example document headi
            self.input_dropdown.addItems(list_of_doc_headings)  # Example values 0-9
            self.input_dropdown.show()
        elif task == "Find similar patents":
            self.input_label.setText("Enter your Text:")
            self.input_textbox.show()
            self.input_dropdown.hide()

    def run_task(self):
        task = self.task_dropdown.currentText()
        article_type = self.article_mapping[self.article_dropdown.currentText()]

        if task == "RAG":
            user_input = self.input_textbox.toPlainText().strip()
            if not user_input:
                QMessageBox.warning(self, "Input Required", "Please enter a query.")
                return
            retriever = RAG_Retriever(self.gen_model_name, self.tokenizer, self.model, 3)
            response = retriever.get_query_answer([article_type], user_input, "RAG")
            # response = f"[Mocked RAG Response for query: {user_input}]"
            self.output_textbox.setPlainText(response)

        elif task == "Extract and summarize information":
            try:
                doc_index = int(self.input_dropdown.currentText())
                extractor = PatentExtractor(self.gen_model_name, self.tokenizer, self.model)
                results = extractor.process_patents(article_type, doc_index)
                # results = [f"[Mocked summary for document index {doc_index}]"]
                self.output_textbox.setPlainText("\n\n".join(results))
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please select a valid document index.")

        elif task == "Find similar patents":
            user_input = self.input_textbox.toPlainText().strip()
            if not user_input:
                QMessageBox.warning(self, "Input Required", "Please enter text.")
                return
            retriever = RAG_Retriever(self.gen_model_name, self.tokenizer, self.model, 1)
            similarity_finder = PatentSimilarityChecker(retriever, self.tokenizer, self.model)
            result = similarity_finder.find_similar_patents(user_input, article_type)
            # result = f"[Mocked Similarity Result for text: {user_input}]"
            self.output_textbox.setPlainText(result)

# Run the Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RAGInterface()
    window.show()
    sys.exit(app.exec())



# ##Code for CLI mode
# from rag_implementation import RAG_Retriever
# from loading_resources import Loader
# from extract_summarize import PatentExtractor
# from similarity_checker import PatentSimilarityChecker

# def main():
#     gen_model_name, tokenizer, model = Loader.load_models()

#     while True:
#         try:
#             task = input("Select the task number to perform: \n1. RAG \n2. Extract and summarize information\n3. Find similar patents\n4. Exit\nEnter choice: ").strip()
            
#             if task == "4" or task.lower() == "exit":
#                 print("Exiting...")
#                 break
            
#             elif task == "1" or task == "2" or task == "3":
#                 article_types = ["a: Human Necessities", "b: Performing Operations; Transporting", 
#                                 "c: Chemistry; Metallurgy", "d: Textiles; Paper", "e: Fixed Constructions", 
#                                 "f: Mechanical Engineering; Lightning; Heating; Weapons; Blasting", "g: Physics", 
#                                 "h: Electricity", "y: General tagging of new or cross-sectional technology"]
                
#                 for idx, article in enumerate(article_types, start=1):
#                     print(f"{idx}. {article}")

#                 choice = input("\nSelect an article type (1-9) or type 'exit' to quit: ").strip()
#                 if choice.lower() == "exit":
#                     print("Exiting...")
#                     break
#                 try:
#                     choice = int(choice) - 1
#                     if choice < 0 or choice >= len(article_types):
#                         raise ValueError("Invalid choice")
#                     article_type = article_types[choice]
#                 except ValueError as e:
#                     print(f"Invalid selection: {e}. Try again.")
#                     continue

#                 if task == "1":
#                     print("Welcome to the RAG Interface (CLI Mode)")
#                     user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
                
#                     if user_query.lower() == "exit":
#                         print("Exiting...")
#                         break
#                     if not user_query:
#                         print("Error: Query cannot be empty.")
#                         continue
                    
#                     print("\nProcessing your query...\n")
#                     try:
#                         response = RAG_Retriever(gen_model_name, tokenizer, model,3).get_query_answer([str(article_type)], str(user_query), str("RAG"))
#                         print("\nRAG Retriever Output:\n" + response)
#                     except Exception as e:
#                         print(f"Error while processing query: {e}")

#                 elif task == "2":
#                     document_index = input("\nSelect the document index which you want information for (or type 'exit' to quit): ").strip()
#                     if document_index.lower() == "exit":
#                         print("Exiting...")
#                         break
#                     try:
#                         document_index = int(document_index)
#                     except ValueError:
#                         print("Invalid document index. Please enter a valid number.")
#                         continue
                
#                     try:
#                         extractor = PatentExtractor(gen_model_name, tokenizer, model)
#                         results = extractor.process_patents(article_type, document_index)
#                         print('Extract and Summarize Information Output:\n')
#                         for result in results:
#                             print(result)
#                     except Exception as e:
#                         print(f"Error while processing patent extraction: {e}")
                    
#                 elif task == "3":
#                     full_text = input("\nGive the content of the new Patent filing (or type 'exit' to quit): ")
#                     if full_text.lower() == "exit":
#                         print("Exiting...")
#                         break
#                     try:
#                         full_text = str(full_text)
#                     except ValueError:
#                         print("Invalid document index. Please enter a valid text.")
#                         continue
                
#                     try:
#                         retriever = RAG_Retriever(gen_model_name, tokenizer, model,1)
#                         similarity_finder = PatentSimilarityChecker(retriever, tokenizer, model)
#                         results = similarity_finder.find_similar_patents(full_text,article_type)
#                         print('Find Similar Patents:\n',results)
#                     except Exception as e:
#                         print(f"Error while processing patent extraction: {e}")

#             else:
#                 print("Invalid input. Please enter 1, 2, or 3 (exit).")
#                 continue
        
#         except Exception as e:
#             print(f"Unexpected error: {e}")
#             continue

# if __name__ == "__main__":
#     main()
