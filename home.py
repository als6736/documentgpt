from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):# llm이 언제 token을 생성하는지
        self.message_box = st.empty()#llm이 token을 생성하기 시작하면 화면에 empty box를 생성

    def on_llm_end(self, *args, **kwargs):# llm이 언제 작업을 끝내는지
        save_message(self.message, "ai")# 작업완료시 메시지 저장

    def on_llm_new_token(self, token, *args, **kwargs):# llm이 언제 new token을 생성하는지
        self.message += token# 메시지에 하나씩 토큰들을 실시간을 추가
        self.message_box.markdown(self.message) #이후 업데이트


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner="Embedding file...")
#computation을 cache. 질문을 입력할때마다 파일을 다시 읽지 않기하기위해
#입력이 변경되지 않는한, 동일한 파일을 함수에 계속 보내면 함수가 다시 실행되지 않는다
def embed_file(file): #파일 임베딩
    file_content = file.read()#파일을 읽고 복사하고
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    #splitter를 생성
    loader = UnstructuredFileLoader(file_path)
    #다양한 파일을 로드하는 함수
    docs = loader.load_and_split(text_splitter=splitter)
    #여러개의 document로 나눠 성능을 높이고 api이용에 드는 비용을 더욱 줄임
    embeddings = OpenAIEmbeddings()
    #openai 임베딩모델 사용
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # CacheBackedEmbeddings로만들어진 embedding을 cache로 저장
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    #docs, cached_embeddings 함께 from_documents메서드 호출 document별로 embedding작업후 결과를 저장한 vector store를 반환 이를 이용하여 document 검색및 연관성이 높은 document들을 찾기도함
    retriever = vectorstore.as_retriever()
    return retriever# retriever로 변경 chain에서 사용할수 있도록
    #retriever는 documents를 제공하는것


def save_message(message, role): #message 딕셔너리 더하는 function
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True): #message send function
    with st.chat_message(role):# role에는 AI나 human이 올수 있고,
        st.markdown(message)
    if save:
        save_message(message, role)# 선택적으로 message를 cache에 save할수 있음


def paint_history():# message store의 모든 메시지에 대해서 send message를 통해 화면에 메시지 출력
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False, #그 메시지를 저장하지는 않음, 이미 저장된 것이기 때문에
        )


def format_docs(docs):#\n 줄바꿈 문자로 구분된 하나의 string으로 합쳐짐
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:#스트림릿을 사용한 사이드바
    file = st.file_uploader(#파일 업로드 요청, 처음에는 파일이 없음
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file: #유저가 파일을 업로드하면 그 파일로 부터 retriever를 얻는다
    retriever = embed_file(file)
    #파일을 임베딩한 뒤에 send message function을 실행한다
    send_message("I'm ready! Ask away!", "ai", save=False)
    #send message function은 메시지를 화면에 표시하고 캐시에 저장
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:# 사용자가 메시지를 보낼시 human으로 보냄
        send_message(message, "human")
        chain = (
            {   #retiever는 document의 list 제공
                "context": retriever | RunnableLambda(format_docs),
                #document list를  format_docs에 제공
                "question": RunnablePassthrough(),
                #체인을 invoke 하고 이곳에 어떤 질문을하면 수정되지 않고 prompt로 이동되길 바람
            }#여기까지가 prompt에대한 input
            | prompt#는 doument와 질문을 설정하고 그것을 llm모델로 보냄
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)


else: 
    st.session_state["messages"] = []
    #session state에 있는 메시지 저장소를 empty list로 초기화