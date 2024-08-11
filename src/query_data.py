from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaEmbeddings

import getpass
import os
#from langchain_openai import OpenAIEmbeddings
#os.environ["OPENAI_API_KEY"] = getpass.getpass()


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama3")
    # = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

def main():
    while True:
        # Prompt the user for input.
        query_text = input("Enter your query (type 'quit' to exit): ")
        if query_text.lower() == 'quit':
            print("Exiting...")
            break
        
        response_text = query_rag(query_text)
        print(f"Response: {response_text}")

def query_rag(query_text: str):
    try:
        print("Preparing the database...")
        # Prepare the DB.
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        print("Searching the database...")
        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        # Extract context and format prompt.
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        print(f"Prompt prepared: {prompt}")

        # Initialize model and get response.
        model = Ollama(model="llama3")
        print("Getting response from model...")
        response_text = model.invoke(prompt)

        # Format and print the response.
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return formatted_response

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    main()
