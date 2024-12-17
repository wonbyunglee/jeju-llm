import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 파일 경로 설정
# 다음과 같이 파일 생성 후 streamlit 구현 코드 실행 시 './data', './modules' 폴더에 옮겨서 실행
file_path = 'alphastorm_data.csv'
faiss_index_path = "alphastorm_faiss.index"
document_ids_path = "alphastorm_documents_ids.npy"

# CSV 파일 불러오기
data = pd.read_csv(file_path)
print("CSV 파일 로드 완료")

# 고유 ID 할당
data['document_id'] = range(len(data))

# Tokenizer와 모델 초기화 (GPU로 이동)
tokenizer = AutoTokenizer.from_pretrained("upskyy/bge-m3-korean")
model = AutoModel.from_pretrained("upskyy/bge-m3-korean").cuda()
print("Tokenizer와 모델 초기화 완료 (GPU 사용)")

# 배치 설정
batch_size = 64
chunk_texts = []
chunk_document_ids = []

# 각 chunk별 텍스트와 해당 문서의 ID를 수집
for idx, row in data.iterrows():
    for col in ["업종명", "주소", "기준연월", "이용건수구간",'이용금액구간', "요일별", "시간대별", "성별", "연령별", "현지인",'url']:
        chunk_text = row[col]
        if pd.notna(chunk_text):  # 유효한 텍스트만 추가
            chunk_texts.append(chunk_text)
            chunk_document_ids.append(row['document_id'])

# DataLoader로 chunk 단위 배치 처리
dataloader = DataLoader(chunk_texts, batch_size=batch_size)
embeddings = []

# 배치별로 임베딩 생성
for batch in tqdm(dataloader, desc="Embedding Chunks"):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).to('cuda')
    embeddings.append(batch_embeddings)

# 임베딩을 하나의 텐서로 변환 후 넘파이 배열로 저장
embeddings_tensor = torch.cat(embeddings, dim=0)

# PyTorch 텐서를 NumPy 배열로 변환
embeddings_np = embeddings_tensor.cpu().numpy()

# FAISS 인덱스 설정 및 임베딩 추가
dimension = embeddings_tensor.shape[1]
index = faiss.IndexFlatIP(dimension)

# L2 정규화 (임베딩을 GPU에서 정규화)
faiss.normalize_L2(embeddings_np)

# FAISS 인덱스에 임베딩 추가
index.add(embeddings_np)

# document_id 저장
np.save(document_ids_path, chunk_document_ids)
faiss.write_index(index, faiss_index_path)  # 다시 CPU로 내보내서 저장
print("FAISS 인덱스 및 chunk별 document_id 저장 완료")