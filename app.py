# %%
import os
import pinecone
import gradio as gr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

embeddings = OpenAIEmbeddings()

# %%
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = "linuxtips"
index = pinecone.Index(index_name)

llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=0)

template="""The Assistant is a legal AI that answers questions.
            The Assistant provides simplified answers based on the provided context.
            The Assistant provides references extracted from the context below. Do not generate additional links or references.
            At the end of the answer, display the extracted references in list format.
            If you cannot find the answer in the context below, or if the question is not related to the legal context, simply say 'I don't know!'

    Question: {query}

    Context: {context}
"""

# %%
prompt = PromptTemplate(
    template=template,
    input_variables=["query", "context"]
)

def search(query):
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    docs = docsearch.similarity_search(query, k=3)
    context = docs[0].page_content + docs[1].page_content + docs[2].page_content
    resp = LLMChain(prompt=prompt, llm=llm)
    return resp.run(query=query, context=context)

with gr.Blocks(title="AI Legal", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# I am an AI whose knowledge base is based on the labor law.")
    query = gr.Textbox(label='Ask your question:', placeholder="For example, how does employee vacation time work?")
    text_output = gr.Textbox(label="Answer")
    btn = gr.Button("Ask")
    btn.click(fn=search, inputs=query, outputs=[text_output])
ui.launch(debug=True, share=True)

# %%
