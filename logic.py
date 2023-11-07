
import os
import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import fitz
from PIL import Image
import openai


COUNT, N = 0, 0
chat_history = []
chain = None  # Initialize 
enable_box = gr.Textbox(value=None,
                      placeholder='Upload your OpenAI API key',
                      interactive=True)
disable_box = gr.Textbox(value='OpenAI API key is Set', interactive=False)

# set the OpenAI API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return disable_box

# enable the API key input box
def enable_api_box():
    return enable_box

# add text to the chat history
def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history.append((text, ''))
    return history

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(pdf_file)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    return text

# process the PDF file and create a conversation chain
def process_file(file):
    if 'OPENAI_API_KEY' not in os.environ:
        raise gr.Error('Upload your OpenAI API key')

    loader = PyPDFLoader(file.name)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()

    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0),
                                   retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                   return_source_documents=True)
    return chain

# generate a response based on the chat history and query
def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain

    if not btn:
        raise gr.Error(message='Upload a PDF')
    if COUNT == 0:   # pdf processing
        chain = process_file(btn)
        COUNT += 1

    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history.append((query, result["answer"]))
    N = list(result['source_documents'][0])[1][1]['page']  # page number of doc

    for char in result['answer']:
        history[-1] = (history[-1][0], history[-1][1] + char)

    yield history, ''

# render a specific page of a PDF file as an image
def render_file(file):
    global N
    doc = fitz.open(file.name)
    page = doc[N]
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

# generate a text summary using ChatGPT API
def generate_summary(text):
    if 'OPENAI_API_KEY' not in os.environ:
        raise gr.Error('Upload your OpenAI API key for summarization')

    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        max_tokens=100  
    )
    return response.choices[0].text











# N appears to be used to store or update information related to the page number of the PDF document.


