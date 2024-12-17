import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss
import streamlit as st
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
import google.generativeai as genai

# 경로 설정
data_path = './data'
module_path = './modules'

# Gemini API 키를 st.secrets에서 가져오기
gemini_api_key = st.secrets["general"]["gemini_api_key"]

# Gemini API 키로 설정
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# 파일 경로 설정
faiss_index_path = os.path.join(module_path, "alphastorm_faiss.index")
document_ids_path = os.path.join(module_path, "alphastorm_documents_ids.npy")
file_path = os.path.join(data_path, 'alphastorm_data.csv')
user_dictionary_path = os.path.join(module_path, 'alphastorm_user_dictionary.txt')  # 사용자 사전 파일 경로

####------------UI-------------###
# Streamlit App UI
st.set_page_config(layout="wide", page_title="🍽️좀 색다른 맛집을 찾는다구? 우리가 해결해줄게!🍽️")

import streamlit as st

# ------------------------전체 배경/글꼴------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jua&display=swap');
    * {
        font-family: 'Jua', sans-serif;
    }

    .stApp {
        background-color: #fdd8b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------타이틀------------------------
st.markdown(
    """
    <h1 style='font-family: "Jua", sans-serif;'>🍊 나만의 제주도 맛집 여행 🍊</h1>
    """,
    unsafe_allow_html=True
)

# ------------------------앱 소개------------------------
with st.expander('다들 모르는 제주도 맛집, 궁금하지 않으세요? 👀'):
    st.markdown(
    """
    <style>
    .stExpander {
        background-color: #faded0;
        padding: 5px;  /* 패딩 추가 */
        border-radius: 10px;  /* 모서리를 둥글게 */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <h3 style='font-family: "Jua", sans-serif;'>남들은 정말 운이 좋은 걸까요? 나는 왜 매번 이렇게 긴 웨이팅에 지칠까요?</h3>
    """,
    unsafe_allow_html=True
    )
    st.write('기대감을 안고 어렵게 한 입 맛봤지만, 그저 그런 맛이었을 때... 모두 이러한 경험이 있지 않으신가요? 😢')
    st.write('저희 앱은 그런 실망스러운 경험 대신, **💎숨겨진 보석 같은 맛집💎** 을 추천해 드립니다.')
    st.write('유명한 음식점들에 가려져 있던 **제주도의 진짜 맛집**, 그곳을 저희가 알려드릴게요! 🍽️💖')

cols_header = st.columns([6, 1])  # 비율을 설정하여 레이블과 초기화 버튼을 나란히 배치

with cols_header[0]:  # 첫 번째 열에 레이블 표시
    st.markdown(
        """
        <h4 style='font-family: "Jua", sans-serif;'>🔎 장소와 음식 종류를 입력해보세요!</h4>
        """,
        unsafe_allow_html=True
    )

# 데이터 및 인덱스 로드
def load_data_and_index():
    if "data_loaded" not in st.session_state:
        # CSV 파일 로드
        data = pd.read_csv(file_path)
        
        # 'complete_document' 열 생성 (여러 열을 결합하여 문서 생성)
        columns_to_combine = ["업종명", "주소", "이용건수구간",'이용금액구간', "요일별", "시간대별", "성별", "연령별", "현지인", 'url']
        data['complete_document'] = data[columns_to_combine].apply(lambda row: ' '.join(row.dropna()), axis=1)
        st.session_state.chunk_texts = data['complete_document'].tolist()
        print("complete_document 열 생성 완료")
        
        # document_ids 로드
        st.session_state.chunk_document_ids = np.load(document_ids_path, allow_pickle=True).tolist()
        print("Document IDs 로드 완료")
        
        # FAISS 인덱스 로드
        st.session_state.index = faiss.read_index(faiss_index_path)
        print("FAISS 인덱스 로드 완료")
        
        # Kiwi 형태소 분석기와 사용자 사전 로드
        kiwi = Kiwi()
        added_words_count = kiwi.load_user_dictionary(user_dictionary_path)
        print(f"사용자 정의 사전에서 {added_words_count}개의 단어가 추가되었습니다.")
        st.session_state.kiwi = kiwi
        
        # BM25에 사용할 코퍼스 토큰화 및 BM25 모델 생성
        tokenized_corpus = [[token.form for token in kiwi.tokenize(doc)] for doc in st.session_state.chunk_texts]
        st.session_state.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 모델 생성 완료")

        # BGE 토크나이저 및 모델 설정 (CPU에서 실행)
        try:
            st.session_state.tokenizer_bge = AutoTokenizer.from_pretrained("upskyy/bge-m3-korean")
            st.session_state.model_bge = AutoModel.from_pretrained("upskyy/bge-m3-korean")  # GPU 사용 X (CPU에서 실행)
            print("BGE 토크나이저 및 모델 설정 완료")
        except Exception as e:
            print(f"토크나이저 로드 중 오류 발생: {e}")
        
        st.session_state.data_loaded = True  # 데이터가 로드된 상태로 플래그 변경
    else:
        print("데이터 및 인덱스가 이미 로드되었습니다.")

# BM25 + FAISS 병렬 검색 함수 (세션 상태에 저장된 데이터 활용)
def search_documents(query, bm25_k=500, faiss_k=10):
    # 1. Kiwi를 이용한 BM25 1차 검색
    tokenized_query_kiwi = [token.form for token in st.session_state.kiwi.tokenize(query)]
    bm25_results = st.session_state.bm25.get_top_n(tokenized_query_kiwi, st.session_state.chunk_texts, n=bm25_k)
    
    if not bm25_results:
        print("BM25 검색 결과가 없습니다.")
        return []  # 빈 결과 반환
    
    # 2. BM25 필터링된 문서들의 document_id 추출
    bm25_filtered_ids = [st.session_state.chunk_document_ids[st.session_state.chunk_texts.index(doc)] for doc in bm25_results]
    
    # 3. BGE 토크나이저로 쿼리 임베딩 생성 (병렬 처리)
    query_inputs_bge = st.session_state.tokenizer_bge(query, return_tensors="pt", padding=True, truncation=True)  # CPU에서 실행
    with torch.no_grad():
        query_outputs_bge = st.session_state.model_bge(**query_inputs_bge)
        query_embedding_bge = query_outputs_bge.last_hidden_state.mean(dim=1).numpy()  # GPU 제거 (CPU에서 실행)
    
    # 4. FAISS 인덱스에서 필터링된 문서들의 임베딩 검색
    faiss.normalize_L2(query_embedding_bge)  # Normalize query embedding for cosine similarity
    
    # FAISS 검색 수행
    distances, indices = st.session_state.index.search(query_embedding_bge, faiss_k)
    
    if indices.shape[0] == 0 or indices[0].size == 0:
        print("FAISS 검색 결과가 없습니다.")
        return []  # 빈 결과 반환
    
    print(f"FAISS 검색 완료 - 상위 {faiss_k}개의 검색 결과:")
    
    # FAISS에서 찾은 상위 문서들의 document_id 추출
    retrieved_document_ids = [st.session_state.chunk_document_ids[idx] for idx in indices[0] if idx < len(st.session_state.chunk_document_ids)]
    unique_document_ids = list(dict.fromkeys(retrieved_document_ids))  # 중복 제거
    
    if not unique_document_ids:
        print("문서 ID가 없습니다.")
        return []
    
    # 최종 검색된 문서들 반환
    retrieved_documents = [st.session_state.chunk_texts[doc_id] for doc_id in unique_document_ids]
    
    # 검색된 문서 출력
    for i, doc in enumerate(retrieved_documents, 1):
        print(f"Document {i}: {doc}")

    return retrieved_documents

# LLM(Gemini) 응답 생성 함수
def generate_gemini_response(query, retrieved_documents):
    # 검색된 문서들을 프롬프트로 결합
    context = " ".join(retrieved_documents)
    
    # 모델에 전달할 최종 프롬프트 구성
    prompt = (
    f"As an AI assistant specializing in tourism in Jeju Island, "
    f"please use the following information to respond to the query:\n{context}\n\n"
    
    f"Query: '{query}'\n\n"
    
    # Example 1: 애월에서 30대 여성이 많이 가는 단품요리 전문점
    f"<Example 1>\n"
    f"User asks: '애월에서 30대 여성이 많이 가는 단품요리 전문점 알려줘'\n\n"
    
    f"Search Results (BM25 and FAISS):\n"
    f"Document1: 간장을 품은 소라게는 애월읍에 위치한 단품요리 전문점입니다. "
    f"이용건수구간은 1구간으로 전체 이용 건수 중 하위 90%에 해당합니다. "
    f"여성 고객 비율이 52.03%로 더 높고, 30대 방문객 비율은 28.3%로 가장 많습니다. "
    f"URL: https://pcmap.place.naver.com/place/1307251244/home\n"
    
    f"Document2: 우미노식탁은 애월읍에 위치한 단품요리 전문점입니다. "
    f"이용건수구간은 2구간으로 전체 이용 건수 중 하위 75%~90%에 해당합니다. "
    f"여성 고객 비율이 55.39%로 더 높고, 30대 방문객 비율은 38.32%로 가장 많습니다. "
    f"URL: https://pcmap.place.naver.com/place/1119656505/home\n"
    
    f"Document3: 제주기와는 애월읍에 위치한 단품요리 전문점입니다. "
    f"이용건수구간은 3구간으로 하위 50%~75%에 해당합니다. "
    f"여성 고객 비율이 56.70%로 더 높고, 30대 방문객 비율은 34.6%로 가장 많습니다. "
    f"URL: https://pcmap.place.naver.com/place/1364855325/home\n\n"
    
    f"Based on these search results:\n"
    
    # 핫플레이스 예시
    f"[🔥 hot place]:\n"
    f"The first recommendation is '간장을 품은 소라게'. It is the most popular among women of age thirties, "
    f"with a usage count range of 1 and a high female visitor percentage of 52.03%. "
    f"URL: https://pcmap.place.naver.com/place/1307251244/home\n\n"
    
    # 콜드플레이스 예시
    f"[💧 cool place]:\n"
    f"A secondary recommendation is '제주기와', which is less crowded compared to [🔥 hot place], "
    f"but still popular among women of age thirties (34.6% female visitors). This makes it a good option for shorter wait times. "
    f"URL: https://pcmap.place.naver.com/place/1364855325/home\n\n"
    
    f"<End of Example 1>\n\n"

    # Example 2: 한립읍에서 점심에 많이 가는 가정식집
    f"<Example 2>\n"
    f"User asks: '한립읍에서 점심에 많이 가는 가정식집을 알려줘'\n\n"
    
    f"Search Results (BM25 and FAISS):\n"
    f"Document1: 양배추식당은 한림읍에 위치한 가정식집입니다. "
    f"이용건수구간은 1구간으로 전체 이용 건수 중 하위 90%에 해당합니다. "
    f"점심 시간(12시~13시)에는 이용객의 95.24%가 방문하며, 남성 고객이 93.93%로 많습니다. "
    f"URL이 없습니다.\n"
    
    f"Document2: 바르왓은 한림읍에 위치한 가정식집입니다. "
    f"이용건수구간은 2구간으로 전체 이용 건수 중 하위 75%~90%에 해당합니다. "
    f"점심 시간(12시~13시)에는 이용객의 53.38%가 방문하며, 여성 고객 비율이 53.02%로 더 높습니다. "
    f"URL: https://pcmap.place.naver.com/place/1691872277/home\n"
    
    f"Document3: 상명식당은 한림읍에 위치한 가정식집입니다. "
    f"이용건수구간은 4구간으로 하위 25%~50%에 해당합니다. "
    f"점심 시간(12시~13시)에는 이용객의 51.07%가 방문하며, 남성 고객 비율이 65.24%로 더 많습니다. "
    f"URL: https://pcmap.place.naver.com/place/1665306171/home\n\n"
    
    f"Based on these search results:\n"
    
    # 핫플레이스 예시
    f"[🔥 hot place]:\n"
    f"The first recommendation is '양배추식당'. It is very popular during lunch time with 95.24% of its visitors coming between 12-1 PM. "
    f"Since it's crowded during lunch, I recommend visiting at other times. No URL is available.\n\n"
    
    # 콜드플레이스 예시
    f"[💧 cool place]:\n"
    f"A secondary recommendation is '바르왓', which is less crowded during lunch but still offers a good dining experience. "
    f"This makes it a quieter option compared to '양배추식당' which is selected as [🔥 hot place]'. "
    f"URL: https://pcmap.place.naver.com/place/1691872277/home\n\n"

    
    f"<End of Example 2>\n\n"

    # Definitions of '이용건수구간' and '이용금액구간'
    f"Definitions of '이용건수구간' and '이용금액구간':\n\n"
    
    # 이용건수구간 설명
    f"'이용건수구간' (Usage Count Range):\n"
    f"1st range: Lower 90% of usage, Higher 10% and above of usage, very crowded\n"
    f"2nd range: Lower 75%-90% of usage, Higher 10%-25% of usage, normally crowded\n"
    f"3rd range: Lower 50%-75% of usage, Higher 25%-50% of usage\n"
    f"4th range: Lower 25%-50% of usage, Higher 50%-75% of usage, not crowded\n"
    f"5th range: Lower 10%-25% of usage, Higher 75%-90% of usage,not crowded\n"
    f"6th range: Lower 10% and above of usage, Higher 90% of usage, not crowded\n\n"

    # 이용금액구간 설명
    f"'이용금액구간' (Spending Range):\n"
    f"1st range: Lower 90% of usage, Higher 10% and above of usage\n"
    f"2nd range: Lower 75%-90% of usage, Higher 10%-25% of usage\n"
    f"3rd range: Lower 50%-75% of usage, Higher 25%-50% of usage\n"
    f"4th range: Lower 25%-50% of usage, Higher 50%-75% of usage\n"
    f"5th range: Lower 10%-25% of usage, Higher 75%-90% of usagen"
    f"6th range: Lower 10% and above of usage, Higher 90% of usage\n\n"

    f"Note: You must provide [🔥hot place] and [💧cool place] as an output followed by the general instruction.\n\n"
    
    # 핫 플레이스 추천
    f"[🔥hot place]:\n\n"
    # f"From the search results, recommend the first place where the majority of visitors are women and age thirties. "
    f"From the search results, recommend the the highest-rated place that satisfies the query: {query}. "
    f"Include the URL directly after the restaurant's description, if available.\n\n"
    
    # 쿨 플레이스 추천
    f"[💧cool place]:\n\n"
    f"From the given search results, recommend a place with similar characteristics, but less crowded compared to [🔥hot place]. "
    f"Explain why it's a quieter or less popular option, based on factors such as time of day, visitor demographics (gender, age), or local resident ratio. "
    f"Make sure to provide reasons why it could be a good alternative to the hot place, such as shorter wait times. "
    f"Include the URL directly after the restaurant's description, if available.\n\n"

    f"Note: The higher the percentage in the 'lower' ranges, the more visitors a place tends to attract.\n\n"
    f"Note: These ranges should be used internally to decide which places to recommend, but do not mention these ranges explicitly in the output.\n\n"
    f"Note: You must answer in korean only.\n\n"
)

    # Gemini 모델 초기화 및 응답 생성
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    return response.text

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []  # messages 리스트를 초기화
    
# 데이터 로드
load_data_and_index()

# 기존 메시지 출력
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    
    if message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(message["content"])

# 새로운 입력을 위해 항상 입력 필드를 대화의 마지막에 유지
placeholder = st.empty()

with placeholder.form(key=f"chat_form_{len(st.session_state.messages)}"):
        query = st.text_input("어디서 무엇을 먹고 싶으신가요? 🤔", "")
        cols = st.columns([0.012,0.3])
        with cols[0]:
            submit = st.form_submit_button("전송")
        with cols[1]:
            reset = st.form_submit_button("처음부터 다시 🗑️")
            
        
# 입력을 받으면 처리
if submit and query:
    st.session_state.messages.append({"role": "user", "content": query})

    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.write(query)
    
    with st.spinner("맛집을 찾고 있어요! 🧐🍜🍲🍗🍣🥯"):
        retrieved_documents = search_documents(query)
        full_response = generate_gemini_response(query, retrieved_documents)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        with st.chat_message("assistant"):
            st.write(full_response)
            
    # 입력 필드가 대화 끝에 계속 유지됨
    placeholder.empty()  # 이전 폼 삭제
    with st.form(key=f"chat_form_{len(st.session_state.messages)}"):
        query = st.text_input("어디서 무엇을 먹고 싶으신가요? 🤔", "")
        cols = st.columns([0.012,0.3])
        with cols[0]:
            submit = st.form_submit_button("전송")
        with cols[1]:
            reset = st.form_submit_button("처음부터 다시 🗑️")

if reset:
    st.session_state.messages = []
    st.session_state["reload"] = True
        