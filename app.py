
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter
from pydantic import BaseModel, Field
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
import os
import re
import json
import uuid
import random
import hashlib
import tempfile
import threading
import warnings

warnings.filterwarnings(action="ignore")
try:
    import pymupdf
    pymupdf.set_messages(path=os.devnull)   # PDF 읽을 때 경고 메시지 감추기 (노트북과 동일)
except Exception:
    pass

# =====================================================================
# AIVLE School 학습 도우미 v5
#
# [필수] 1 PDF 업로드(드래그 + '+'버튼) / 2 RAG 채팅(rag_pipeline) / 3 답변 출처 표시
#        - 교안 44쪽 배치, 46쪽 주의 사항(체인은 session_state, 파일 "경로" 전달) 유지
# [추가] 4 FAQ·추천 질문 / 5 대화 기록 관리 / 8 백서 기반 학습 계획 / 9 이해도 퀴즈
#        11 관리자 전용 페이지 / 12 질문 통계 / 13 담당자 문의 / 14 UI / 15 문서 관리
#
# 원칙: 모든 AI 기능은 "백서에 있는 내용만" 근거로 답한다. (필수 기능 4)
# =====================================================================

APP_VERSION = "v5 · 12개 기능 + 백서 기반 추천 질문"   # 배포 확인용 (사이드바 맨 아래 표시)
LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
NO_ANSWER = "백서에서 해당 내용을 확인할 수 없습니다."
KST = ZoneInfo("Asia/Seoul")

DATA_DIR = Path("data")            # 문서·통계·문의 저장 위치 (앱이 재시작되면 초기화될 수 있음)
DOC_DIR = DATA_DIR / "docs"
BUNDLED_DIR = Path("whitepaper")   # GitHub 저장소의 whitepaper 폴더에 PDF를 넣으면 기본 문서로 자동 등록

DEFAULT_CONFIG = {
    "settings": {"k": 4, "chunk_size": 1000, "chunk_overlap": 200},
    "questions": [
        "AIVLE School은 어떤 교육 과정인가요?",
        "비전공자는 무엇부터 공부해야 하나요?",
        "하루 공부 시간은 어느 정도가 적당한가요?",
        "프로젝트는 어떻게 준비해야 하나요?",
        "선배들이 꼭 해주고 싶은 조언은 무엇인가요?",
        "취업을 잘하려면 무엇이 중요한가요?",
    ],
    "faq": [
        "AIVLE School은 누구를 위한 교육 과정인가요?",
        "비전공자도 따라갈 수 있나요?",
        "교육은 어떤 방식으로 진행되나요?",
        "미니 프로젝트와 빅 프로젝트는 무엇인가요?",
        "수업 외에 따로 공부해야 할 것이 있나요?",
        "수료 후 취업 준비는 어떻게 하나요?",
    ],
    "default_doc": None,
    "question_source": "auto",   # auto: 백서를 읽고 자동 생성 / manual: 관리자가 직접 입력한 목록
}

INQUIRY_CATEGORIES = ["지원·선발", "교육 과정", "학습 방법", "프로젝트", "취업", "기타"]

SYSTEM_PROMPT = """당신은 AIVLE School DX 트랙 학습 도우미입니다.

아래 [참고 문서]만 근거로 질문에 답하세요.

답변 규칙:
1. 질문에 직접 답하고, 필요한 경우 항목별로 설명하세요.
2. 공식 안내, 개인 후기, 작성자의 학습 제안을 구분하세요.
3. 문서에 근거가 없거나 질문이 문서 내용과 관계없으면 다른 말 없이
   "백서에서 해당 내용을 확인할 수 없습니다."라고만 답하세요.
4. 지원 조건, 일정, 수료 기준처럼 변경될 수 있는 정보는
   해당 기수의 공식 공지 확인이 필요함을 안내하세요.
5. 문서에 없는 사실이나 수치를 만들지 마세요. 일반 상식으로 내용을 보충하지 마세요.
6. 친절하고 이해하기 쉬운 한국어로 답하세요.
7. 규칙 3에 해당하지 않는 답변의 마지막에는, 문서 내용 범위 안에서
   학습자가 바로 실천할 수 있는 응원이나 다음 행동을 한 줄 덧붙이세요.

[참고 문서]
{context}"""

# 학습 계획 등 "정리" 작업용 프롬프트 (채팅과 같은 원칙: 백서 밖 내용 금지)
TASK_SYSTEM = """당신은 AIVLE School DX 트랙 학습 도우미입니다.
아래 [참고 문서]만 근거로 [작업]을 수행하세요.

규칙:
1. 문서에 없는 사실, 수치, 일정, 자료, 활동을 만들지 마세요. 일반 상식으로 보충하지 마세요.
2. 요청 조건에 맞는 내용이 문서에 없으면 그 항목에는 "백서에 해당 내용 없음"이라고 쓰세요.
3. 문서 전체에서 근거를 하나도 찾을 수 없으면 다른 말 없이
   "백서에서 해당 내용을 확인할 수 없습니다."라고만 답하세요.
4. 각 항목 끝에 근거 페이지를 (p.3)처럼 표기하세요. 페이지는 [참고 문서]의 [p.N] 표시에서만 고르세요.
5. 공식 안내, 선배 후기, 작성자의 학습 제안을 구분하세요.
6. 일정·지원 조건처럼 바뀔 수 있는 정보는 해당 기수 공식 공지 확인이 필요하다고 안내하세요.
7. 친절하고 이해하기 쉬운 한국어로, 마지막에 문서 범위 안의 응원 한 줄을 덧붙이세요.

[작업]
{task}

[참고 문서]
{context}"""

PLAN_TASK = (
    "백서에 정리된 학습 전략, 학습 계획, 공부 시간, 선배 조언을 모아 사용자의 조건에 맞게 "
    "학습 계획을 표 또는 체크리스트로 정리하세요. 백서에 있는 내용을 고르고 배치하는 것만 허용되며, "
    "백서에 없는 새로운 활동이나 일정은 지어내지 마세요."
)

QUIZ_SYSTEM = """당신은 AIVLE School 백서로 이해도 확인 문제를 만드는 출제자입니다.
아래 [참고 문서]의 내용만으로 4지선다 문제를 {n}개 만드세요.

규칙:
1. 문제, 정답, 해설은 모두 문서에 적힌 내용이어야 합니다. 문서에 없는 사실을 묻거나 만들지 마세요.
2. 보기는 4개이고 정답은 하나뿐이어야 합니다. 오답은 문서 내용과 분명히 구별되게 하세요.
3. answer에는 정답 보기의 번호를 0부터 세어 적으세요.
4. page에는 근거가 된 페이지를 [참고 문서]의 [p.N] 표시에서 골라 "p.N" 형식으로 적으세요.
5. 쉬운 한국어로 쓰세요.

[참고 문서]
{context}"""

SUGGEST_SYSTEM = """당신은 AIVLE School 백서를 읽고 예비·현재 교육생에게 보여 줄 추천 질문을 만드는 담당자입니다.
아래 [참고 문서]를 읽고, 문서 안에 답이 분명히 적혀 있는 질문만 {n}개 만드세요.

규칙:
1. 문서에서 답을 찾을 수 없는 질문은 절대 만들지 마세요.
2. 교육생이 실제로 궁금해할 만한 질문으로, 서로 다른 주제를 고르게 다루세요.
   (예: 과정 소개, FAQ, 선배 후기·조언, 학습 전략, 학습 계획, 프로젝트, 취업 준비 중 문서에 있는 것)
3. 한 문장, 30자 안팎의 자연스러운 한국어 질문으로 쓰고 물음표로 끝내세요.

[참고 문서]
{context}"""

FAQ_SYSTEM = """당신은 AIVLE School 백서에서 FAQ(자주 묻는 질문)를 정리하는 담당자입니다.
아래 [참고 문서]에서 FAQ·자주 묻는 질문·Q&A로 적힌 질문을 찾아 최대 {n}개 뽑으세요.

규칙:
1. 문서에 실제로 적힌 질문을 우선 그대로(필요하면 자연스럽게 다듬어) 옮기세요.
2. 문서에 FAQ 형식이 없으면, 문서 안에 답이 분명히 적힌 질문만 만드세요.
3. 문서에서 답을 찾을 수 없는 질문은 절대 넣지 마세요.
4. 한 문장의 한국어 질문으로 쓰고 물음표로 끝내세요.

[참고 문서]
{context}"""

REWRITE_SYSTEM = (
    "이전 대화를 참고해, 마지막 질문을 이전 대화 없이도 이해되는 독립된 질문 한 문장으로 다시 쓰세요. "
    "질문에 답하지 말고 다시 쓴 질문만 출력하세요. 이미 독립적인 질문이면 그대로 출력하세요."
)

STOPWORDS = {"어떻게", "무엇", "무엇인가요", "있나요", "하나요", "되나요", "알려주세요", "에이블스쿨",
             "AIVLE", "School", "그리고", "어떤", "정도", "대한", "관련", "하면", "해야", "인가요"}


# 문서 포맷팅 함수 (기존 코드 그대로)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================================
# RAG 파이프라인 (01번 노트북 함수와 같은 동작)
# rag_pipeline(경로)만 넣으면 기존과 똑같이 체인을 반환한다.
# 뒤의 매개변수는 관리자 설정·근거 표시용이며 기본값은 01번과 동일하다.
# =====================================================================
def rag_pipeline(filepath, chunk_size=1000, chunk_overlap=200, k=4,
                 collection_name="aivle_docs", return_details=False):
    # 1. PDF 파일 읽기 (비어 있는 문서 제외)
    loader = PyMuPDF4LLMLoader(str(filepath))
    docs = [doc for doc in loader.load() if doc.page_content.strip()]
    if not docs:
        raise ValueError("PDF에서 읽을 수 있는 본문이 없습니다.")
    print(f"1. PDF 읽기 완료: {len(docs)}개 Document")

    # 2. 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"2. 문서 분할 완료: {len(chunks)}개 조각")

    # 3. 임베딩 모델
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. 벡터 저장소 (재실행 시 중복 저장 방지를 위해 초기화)
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings)
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)
    print("3. 임베딩 및 벡터 저장 완료")

    # 5. 검색기
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 6. 프롬프트
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])

    # 7. 답변 모델
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0)

    # 8. RAG 체인 연결
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("4. RAG 체인 구성 완료")

    if return_details:
        return rag_chain, {"retriever": retriever, "pages": len(docs),
                           "chunks": len(chunks), "documents": chunks}
    return rag_chain


# =====================================================================
# 퀴즈 출력 형식
# =====================================================================
class QuizItem(BaseModel):
    question: str = Field(description="문제")
    options: list[str] = Field(description="보기 4개")
    answer: int = Field(description="정답 보기 번호 (0부터 시작)")
    explanation: str = Field(description="정답 해설 (문서 근거)")
    page: str = Field(description="근거 페이지, 예: p.3")


class Quiz(BaseModel):
    items: list[QuizItem]


class QuestionList(BaseModel):
    questions: list[str] = Field(description="백서에서 답을 찾을 수 있는 추천 질문 목록")


# =====================================================================
# 페이지 설정 & 스타일
# =====================================================================
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="📘", layout="wide")

st.markdown("""
<style>
.logo-card {border:1px solid #d0d5da; border-radius:12px; padding:.9rem; background:#f8f9fa;
            text-align:center; margin-bottom:.8rem;}
.logo-card img {width:130px;}
.logo-card p {font-size:.8rem; color:#6c757d; margin:.4rem 0 0 0;}
.header-box {border:1px solid #d0d5da; border-radius:12px; padding:.9rem 1.2rem; background:#f8f9fa;
             display:flex; align-items:center; gap:.8rem; margin-bottom:1rem;}
.header-box h2 {margin:0; padding:0; font-size:1.6rem;}
.header-box p {margin:.1rem 0 0 0; color:#6c757d; font-size:.9rem;}
.st-key-chat_area {border:2px dashed #3aa8a0; border-radius:12px; padding:1rem;}
.hero {background:#3aa8a00d; border:1px solid #3aa8a040; border-radius:12px; padding:1rem 1.2rem; margin-bottom:.8rem;}
.hero h4 {margin:0 0 .3rem 0;}
.badge {display:inline-block; border-radius:999px; padding:.1rem .6rem; font-size:.8rem; background:#e9ecef;}
.badge-ok {background:#d3f9d8; color:#2b8a3e;}
.badge-wait {background:#fff3bf; color:#e67700;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 공용 저장소 (모든 사용자가 함께 쓰는 데이터: 설정·문서·통계·문의)
# =====================================================================
def now_str():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass   # 저장 실패해도 앱은 계속 동작


def md5(data):
    return hashlib.md5(data).hexdigest()


@st.cache_resource
def get_store():
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config.update(read_json(DATA_DIR / "config.json", {}))

    docs = {h: m for h, m in read_json(DATA_DIR / "docs.json", {}).items() if Path(m["path"]).exists()}
    if BUNDLED_DIR.exists():
        for p in sorted(BUNDLED_DIR.glob("*.pdf")):
            docs.setdefault(md5(p.read_bytes()), {
                "name": p.name, "path": str(p), "size": p.stat().st_size,
                "uploaded_at": "저장소 기본 포함", "pages": None, "chunks": None, "bundled": True,
            })
    if config.get("default_doc") not in docs:
        config["default_doc"] = next(iter(docs), None)

    return {
        "lock": threading.Lock(),
        "config": config,
        "docs": docs,
        "logs": read_json(DATA_DIR / "logs.json", []),
        "inquiries": read_json(DATA_DIR / "inquiries.json", []),
        "faq_cache": {},
    }


STORE = get_store()
CFG = STORE["config"]
FILES = {"config": "config.json", "docs": "docs.json", "logs": "logs.json", "inquiries": "inquiries.json"}


def persist(*names):
    for name in names:
        write_json(DATA_DIR / FILES[name], STORE[name])


# =====================================================================
# 세션 상태 초기화
# =====================================================================
defaults = {
    "session_id": uuid.uuid4().hex[:8],
    "rag_chain": None,           # 교안 46쪽: RAG 체인은 세션 상태에 저장
    "retriever": None,
    "doc_chunks": [],
    "doc_hash": None,
    "doc_bytes": None,
    "doc_info": None,
    "manual_unload": False,
    "pending_question": None,
    "goto": None,
    "uploader_key": 0,
    "last_file_id": None,
    "last_lib_sel": None,
    "last_import_id": None,
    "is_admin": False,
    "use_context": True,
    "inquiry_prefill": None,
    "plan": None,
    "quiz": None,
    "admin_upload_key": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# 대화 기록: 여러 대화를 만들고 전환할 수 있음 (메시지는 기존 코드처럼 HumanMessage / AIMessage)
if "conversations" not in st.session_state:
    first_id = uuid.uuid4().hex[:8]
    st.session_state.conversations = {first_id: {"title": "새 대화", "messages": [], "created": now_str()}}
    st.session_state.current_conv = first_id
st.session_state.messages = st.session_state.conversations[st.session_state.current_conv]["messages"]


# =====================================================================
# 보조 함수
# =====================================================================
def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def page_label(doc):
    page = doc.metadata.get("page")
    return f"p.{page + 1}" if isinstance(page, int) else "페이지 정보 없음"


def format_docs_with_pages(docs):
    return "\n\n".join(f"[{page_label(d)}]\n{d.page_content}" for d in docs)


def to_sources(docs):
    return [{"page": page_label(d), "text": d.page_content[:300]} for d in docs]


def render_sources(sources, nested=False):
    """답변 근거 표시. 다른 expander 안에서는 expander를 겹칠 수 없어 간단히 표시한다."""
    if not sources:
        return
    pages = ", ".join(dict.fromkeys(s["page"] for s in sources))
    if nested:
        st.caption(f"📎 답변 근거: {pages}")
        return
    with st.expander(f"📎 답변 근거 ({pages})"):
        for s in sources:
            st.markdown(f"**{s['page']}**")
            st.caption(s["text"])


def doc_name():
    info = st.session_state.doc_info
    return info["name"] if info else None


def log_event(kind, question, answer="", no_answer=False, pages=None, extra=None):
    """모든 사용자의 이용 기록을 공용 저장소에 남긴다 (질문 통계용)."""
    entry = {
        "id": uuid.uuid4().hex[:12], "time": now_str(), "session": st.session_state.session_id,
        "type": kind, "question": question, "answer": (answer or "")[:500],
        "no_answer": bool(no_answer), "pages": pages or [], "doc": doc_name(), "feedback": None,
    }
    entry.update(extra or {})
    with STORE["lock"]:
        STORE["logs"].append(entry)
        del STORE["logs"][:-5000]       # 최근 5000건만 보관
        persist("logs")
    return entry["id"]


def on_feedback(msg_id):
    value = st.session_state.get(f"fb_{msg_id}")
    for m in st.session_state.messages:
        if isinstance(m, AIMessage) and m.additional_kwargs.get("id") == msg_id:
            m.additional_kwargs["feedback"] = value
    with STORE["lock"]:
        for e in STORE["logs"]:
            if e["id"] == msg_id:
                e["feedback"] = value
        persist("logs")


# ---------------------------------------------------------------------
# 문서 → RAG 체인 (교안 46쪽: 임시 .pdf 파일을 만들고 "경로"를 rag_pipeline에 전달)
# 같은 문서·같은 설정이면 모든 사용자가 한 번 만든 체인을 재사용 → 벡터화 반복 없음
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=10)
def build_rag(file_hash, _file_bytes, chunk_size, chunk_overlap, k):
    # ① 임시 파일 생성 (.pdf) ② 업로드 내용 저장 ③ 경로 확보 (tmp.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name
    try:
        # ④ RAG 파이프라인에 "경로" 전달
        return rag_pipeline(
            tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k,
            collection_name=f"aivle_{file_hash[:16]}_{chunk_size}_{chunk_overlap}_{k}",
            return_details=True,
        )
    finally:
        os.remove(tmp_path)


def activate_doc(file_bytes, name, file_hash=None, notify=True):
    file_hash = file_hash or md5(file_bytes)
    s = CFG["settings"]
    try:
        with st.spinner(f"'{name}' 문서를 분석하고 있어요..."):
            chain, details = build_rag(file_hash, file_bytes, s["chunk_size"], s["chunk_overlap"], s["k"])
    except Exception as e:
        st.error(f"문서 처리 실패: {e}")
        return False

    st.session_state.rag_chain = chain
    st.session_state.retriever = details["retriever"]
    st.session_state.doc_chunks = details["documents"]
    st.session_state.doc_hash = file_hash
    st.session_state.doc_bytes = file_bytes
    st.session_state.doc_info = {
        "name": name, "pages": details["pages"], "chunks": details["chunks"],
        "built_at": now_str(), **s,
    }
    st.session_state.manual_unload = False
    with STORE["lock"]:
        if file_hash in STORE["docs"]:
            STORE["docs"][file_hash].update(pages=details["pages"], chunks=details["chunks"])
            persist("docs")
    if notify:
        st.toast(f"✅ '{name}' 준비 완료")
    return True


def load_library_doc(file_hash, notify=True):
    meta = STORE["docs"].get(file_hash)
    if not meta:
        return False
    return activate_doc(Path(meta["path"]).read_bytes(), meta["name"], file_hash, notify)


def add_to_library(file_bytes, name):
    file_hash = md5(file_bytes)
    path = DOC_DIR / f"{file_hash}.pdf"
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file_bytes)
    with STORE["lock"]:
        old = STORE["docs"].get(file_hash, {})
        STORE["docs"][file_hash] = {
            "name": name, "path": str(path), "size": len(file_bytes), "uploaded_at": now_str(),
            "pages": old.get("pages"), "chunks": old.get("chunks"), "bundled": False,
        }
        if not CFG.get("default_doc"):
            CFG["default_doc"] = file_hash
        persist("docs", "config")
    return file_hash


def delete_library_doc(file_hash):
    with STORE["lock"]:
        meta = STORE["docs"].pop(file_hash, None)
        if meta and not meta.get("bundled"):
            Path(meta["path"]).unlink(missing_ok=True)
        if CFG.get("default_doc") == file_hash:
            CFG["default_doc"] = next(iter(STORE["docs"]), None)
        persist("docs", "config")


def unload_document():
    for key in ["rag_chain", "retriever", "doc_hash", "doc_bytes", "doc_info", "last_file_id", "last_lib_sel"]:
        st.session_state[key] = None
    st.session_state.doc_chunks = []
    st.session_state.manual_unload = True
    st.session_state.uploader_key += 1


def multi_retrieve(queries, limit=10):
    """여러 검색어로 백서를 찾아 중복 없이 모은다 (학습 계획·퀴즈용)."""
    found, seen = [], set()
    for q in queries:
        for d in st.session_state.retriever.invoke(q):
            if d.page_content not in seen:
                seen.add(d.page_content)
                found.append(d)
    return found[:limit]


def doc_task_stream(task, request, docs):
    prompt = ChatPromptTemplate.from_messages([("system", TASK_SYSTEM), ("human", "{request}")])
    chain = prompt | ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0) | StrOutputParser()
    return chain.stream({"task": task, "context": format_docs_with_pages(docs), "request": request})


def standalone_question(question, history):
    """후속 질문("그건 어떻게 해?")을 이전 대화를 반영한 독립 질문으로 바꿔 검색 정확도를 높인다.
    답변은 여전히 기존 RAG 체인이 백서만 근거로 생성한다."""
    if not history:
        return question
    lines = []
    for m in history[-4:]:
        who = "사용자" if isinstance(m, HumanMessage) else "도우미"
        lines.append(f"{who}: {m.content[:300]}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM),
        ("human", "[이전 대화]\n{history}\n\n[마지막 질문]\n{question}"),
    ])
    try:
        result = (prompt | get_llm() | StrOutputParser()).invoke(
            {"history": "\n".join(lines), "question": question}).strip()
        return result or question
    except Exception:
        return question


# ---------------------------------------------------------------------
# 백서 기반 추천 질문 자동 생성 (문서마다 한 번만 만들고 모든 사용자가 재사용)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=40)
def suggest_questions(doc_hash, _chunks, kind="recommend", n=6):
    """업로드된 '백서'를 읽고 질문을 만든다. kind: recommend(추천 질문) / faq(백서의 FAQ 부분에서 추출)"""
    if not _chunks:
        return []
    sample = []
    if kind == "faq":   # 백서의 FAQ 부분을 우선 사용
        sample = [d for d in _chunks if re.search(r"FAQ|자주\s*묻는|Q\s*[0-9]|Q\.|Q\s*:", d.page_content)][:12]
    if not sample:      # 문서 앞뒤를 고르게 훑도록 간격을 두고 뽑음
        step = max(1, len(_chunks) // 12)
        sample = _chunks[::step][:12]
    context = "\n\n".join(f"[{page_label(d)}]\n{d.page_content[:800]}" for d in sample)
    system = FAQ_SYSTEM if kind == "faq" else SUGGEST_SYSTEM
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "질문 목록을 만들어 주세요.")])
    result = (prompt | get_llm().with_structured_output(QuestionList)).invoke({"n": n, "context": context})
    questions = [q.strip() for q in result.questions if q.strip()]
    return list(dict.fromkeys(questions))[:n]


def _questions_for(kind, fallback, n):
    """자동 모드면 현재 백서로 만든 질문, 실패하거나 수동 모드면 관리자가 입력한 목록."""
    if CFG.get("question_source") == "auto" and st.session_state.doc_hash:
        try:
            auto = suggest_questions(st.session_state.doc_hash, st.session_state.doc_chunks, kind, n)
            if auto:
                return auto
        except Exception:
            pass
    return fallback


def recommended_questions():
    return _questions_for("recommend", CFG["questions"], 6)


def faq_questions():
    return _questions_for("faq", CFG["faq"], 8)


def normalize_q(q):
    return re.sub(r"[\s?？!.]+$", "", q.strip()).lower()


# ---------------------------------------------------------------------
# 대화 관리
# ---------------------------------------------------------------------
def ask(question):
    st.session_state.pending_question = question
    st.session_state.goto = "chat"


def go_inquiry(question):
    st.session_state.inquiry_prefill = question
    st.session_state.goto = "inquiry"


def go(page_key):
    st.session_state.goto = page_key


def new_conversation():
    cid = uuid.uuid4().hex[:8]
    st.session_state.conversations[cid] = {"title": "새 대화", "messages": [], "created": now_str()}
    st.session_state.current_conv = cid
    st.session_state.conv_select = cid


def delete_conversation():
    convs = st.session_state.conversations
    cur = st.session_state.current_conv
    for m in convs[cur]["messages"]:
        st.session_state.pop(f"fb_{m.additional_kwargs.get('id')}", None)
    if len(convs) > 1:
        del convs[cur]
        st.session_state.current_conv = list(convs)[-1]
    else:
        convs[cur]["messages"].clear()
        convs[cur]["title"] = "새 대화"
    st.session_state.conv_select = st.session_state.current_conv


def switch_conversation():
    st.session_state.current_conv = st.session_state.conv_select


def conversation_to_json(messages):
    return [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content,
             **({"sources": m.additional_kwargs.get("sources", [])} if isinstance(m, AIMessage) else {})}
            for m in messages]


def conversation_to_markdown(messages, title):
    lines = [f"# {title}", ""]
    for m in messages:
        who = "🙋 질문" if isinstance(m, HumanMessage) else "📘 답변"
        lines += [f"### {who}", m.content, ""]
    return "\n".join(lines)


def import_conversation(raw):
    items = json.loads(raw)
    messages = []
    for it in items:
        if it.get("role") == "user":
            messages.append(HumanMessage(content=it.get("content", "")))
        else:
            messages.append(AIMessage(content=it.get("content", ""), additional_kwargs={
                "id": uuid.uuid4().hex[:12], "sources": it.get("sources", []),
                "no_answer": NO_ANSWER in it.get("content", ""), "feedback": None}))
    first_q = next((m.content for m in messages if isinstance(m, HumanMessage)), "불러온 대화")
    cid = uuid.uuid4().hex[:8]
    st.session_state.conversations[cid] = {"title": "📂 " + first_q[:18], "messages": messages, "created": now_str()}
    st.session_state.current_conv = cid
    st.session_state.conv_select = cid


# =====================================================================
# 기본 문서 자동 준비 (관리자가 등록한 기본 문서가 있으면 업로드 없이 바로 사용)
# =====================================================================
if (st.session_state.rag_chain is None and not st.session_state.manual_unload
        and CFG.get("default_doc") in STORE["docs"]):
    load_library_doc(CFG["default_doc"], notify=False)

# 관리자가 RAG 설정을 바꿨으면 현재 문서를 새 설정으로 다시 준비
_info = st.session_state.doc_info
if _info and st.session_state.doc_bytes and any(_info.get(k) != v for k, v in CFG["settings"].items()):
    activate_doc(st.session_state.doc_bytes, _info["name"], st.session_state.doc_hash, notify=False)


# =====================================================================
# 사이드바 (교안 44쪽: 로고 → 관리자(파일 업로드) → 추천 질문 → 대화 삭제)
# =====================================================================
with st.sidebar:
    st.markdown(f"""<div class="logo-card"><img src="{LOGO_URL}">
        <p>에이블스쿨 백서를 바탕으로<br>학습 고민에 답해 드립니다.</p></div>""",
                unsafe_allow_html=True)

    # --- 관리자: 파일 업로드 (드래그 앤 드롭) / 문서 변경 ---
    st.subheader("관리자")
    uploaded = st.file_uploader("백서 PDF를 드래그하거나 클릭해 선택하세요", type=["pdf"],
                                key=f"uploader_{st.session_state.uploader_key}")
    if uploaded is not None:
        file_id = getattr(uploaded, "file_id", None) or f"{uploaded.name}_{uploaded.size}"
        if file_id != st.session_state.last_file_id:      # 새로 올린 경우에만 처리
            st.session_state.last_file_id = file_id
            activate_doc(uploaded.getvalue(), uploaded.name)

    if STORE["docs"]:
        lib_sel = st.selectbox("📚 등록된 문서로 바꾸기", options=list(STORE["docs"]),
                               format_func=lambda h: STORE["docs"][h]["name"], index=None,
                               placeholder="등록된 문서 선택", key=f"lib_{st.session_state.uploader_key}")
        if lib_sel and lib_sel != st.session_state.last_lib_sel:
            st.session_state.last_lib_sel = lib_sel
            load_library_doc(lib_sel)

    info = st.session_state.doc_info
    if info:
        st.success(f"사용 중: **{info['name']}**\n\n{info['pages']}페이지 · {info['chunks']}개 조각")
        st.button("📤 문서 해제", on_click=unload_document, width="stretch")
    else:
        st.info("백서 PDF를 올리면 바로 질문할 수 있어요.")

    # --- 추천 질문 ---
    st.subheader("💡 추천 질문")
    if st.session_state.rag_chain is not None:
        with st.spinner("백서로 추천 질문 만드는 중..."):
            REC_QUESTIONS = recommended_questions()
    else:
        REC_QUESTIONS = CFG["questions"]
    for i, q in enumerate(REC_QUESTIONS):
        st.button(q, key=f"side_q_{i}", on_click=ask, args=(q,), width="stretch",
                  disabled=st.session_state.rag_chain is None)

    # --- 대화 기록 관리 (새 대화 / 전환 / 삭제 / 저장) ---
    st.subheader("💬 대화")
    convs = st.session_state.conversations
    if st.session_state.get("conv_select") not in convs:
        st.session_state.conv_select = st.session_state.current_conv
    if len(convs) > 1:
        st.selectbox("대화 선택", options=list(convs), format_func=lambda c: convs[c]["title"],
                     key="conv_select", on_change=switch_conversation, label_visibility="collapsed")
    c1, c2 = st.columns(2)
    c1.button("➕ 새 대화", on_click=new_conversation, width="stretch",
              disabled=not st.session_state.messages)
    c2.button("🗑️ 대화 삭제", on_click=delete_conversation, width="stretch",
              disabled=not st.session_state.messages and len(convs) == 1)
    st.download_button("💾 대화 저장", conversation_to_markdown(st.session_state.messages,
                                                             convs[st.session_state.current_conv]["title"]),
                       file_name="aivle_chat.md", mime="text/markdown", width="stretch",
                       disabled=not st.session_state.messages)

    # --- 관리자 설정 (비밀번호) ---
    st.divider()
    with st.expander("🔐 관리자 설정", expanded=st.session_state.is_admin):
        if not st.session_state.is_admin:
            pw = st.text_input("비밀번호", type="password")
            if st.button("로그인", width="stretch"):
                if pw == get_secret("ADMIN_PASSWORD", "aivle"):
                    st.session_state.is_admin = True
                    st.session_state.goto = "admin"
                    st.rerun()
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
        else:
            st.success("관리자로 로그인했습니다.")
            st.button("🛠️ 관리자 페이지 열기", on_click=go, args=("admin",), width="stretch")
            if st.button("로그아웃", width="stretch"):
                st.session_state.is_admin = False
                st.rerun()

    st.caption(f"앱 버전: {APP_VERSION}")


# =====================================================================
# 페이지 ① 학습 도우미 (채팅)
# =====================================================================
def page_chat():
    # 상단 도구: 맥락 반영 / 대화 내보내기·불러오기
    t1, t2 = st.columns([3, 1])
    st.session_state.use_context = t1.toggle("이전 대화 맥락 반영 (후속 질문 이해)",
                                             value=st.session_state.use_context)
    with t2.popover("🗂️ 대화 기록", width="stretch"):
        conv = st.session_state.conversations[st.session_state.current_conv]
        st.download_button("JSON으로 내보내기",
                           json.dumps(conversation_to_json(st.session_state.messages), ensure_ascii=False, indent=2),
                           file_name="aivle_chat.json", mime="application/json", width="stretch",
                           disabled=not st.session_state.messages)
        st.download_button("Markdown으로 내보내기", conversation_to_markdown(st.session_state.messages, conv["title"]),
                           file_name="aivle_chat.md", mime="text/markdown", width="stretch",
                           disabled=not st.session_state.messages)
        imported = st.file_uploader("저장한 대화(JSON) 불러오기", type=["json"], key="conv_import")
        if imported is not None:
            fid = getattr(imported, "file_id", None) or imported.name
            if fid != st.session_state.last_import_id:
                st.session_state.last_import_id = fid
                try:
                    import_conversation(imported.getvalue().decode("utf-8"))
                    st.rerun()
                except Exception as e:
                    st.error(f"불러오기 실패: {e}")

    with st.container(key="chat_area"):
        msg_box = st.container(height=520, border=False)    # 대화가 길어지면 이 안에서 스크롤
        value = st.chat_input("질문을 입력하세요  (+ 버튼으로 백서 PDF 첨부)", accept_file=True, file_type=["pdf"])

    user_input, attached = None, None
    if value:
        user_input = (value.text or "").strip() or None
        attached = value.files[0] if value.files else None

    # '+' 버튼으로 첨부한 PDF 처리
    if attached is not None:
        with msg_box:
            activate_doc(attached.getvalue(), attached.name)

    # 추천 질문·FAQ 등에서 넘어온 질문
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    with msg_box:
        # 첫 화면
        if not st.session_state.messages and not user_input:
            if st.session_state.rag_chain:
                st.markdown(f"""<div class="hero"><h4>무엇이 궁금하신가요? 🙌</h4>
                    📄 {doc_name()} · 아래 질문을 눌러 바로 시작해 보세요.</div>""", unsafe_allow_html=True)
                cols = st.columns(2)
                for i, q in enumerate(REC_QUESTIONS[:6]):
                    cols[i % 2].button(f"💡 {q}", key=f"main_q_{i}", on_click=ask, args=(q,), width="stretch")
                st.markdown("**이런 것도 할 수 있어요**")
                a, b, c = st.columns(3)
                a.button("❓ 자주 묻는 질문", on_click=go, args=("faq",), width="stretch")
                b.button("🗓️ 백서 기반 학습 계획", on_click=go, args=("plan",), width="stretch")
                c.button("📝 이해도 퀴즈", on_click=go, args=("quiz",), width="stretch")
            else:
                st.markdown("""<div class="hero"><h4>시작하려면 백서를 올려 주세요 📂</h4>
                    왼쪽 <b>관리자</b> 영역에 PDF를 <b>드래그</b>하거나, 입력창의 <b>+</b> 버튼으로 첨부하세요.</div>""",
                            unsafe_allow_html=True)

        # 기존 대화 내용 출력 (기존 코드 방식)
        msgs = st.session_state.messages
        for i, message in enumerate(msgs):
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role, avatar="🙋" if role == "user" else "📘"):
                st.markdown(message.content)
                if role == "assistant":
                    meta = message.additional_kwargs
                    if meta.get("rewritten"):
                        st.caption(f"🔁 이렇게 이해했어요: {meta['rewritten']}")
                    render_sources(meta.get("sources"))
                    mid = meta.get("id", f"m{i}")
                    if meta.get("no_answer"):
                        prev_q = msgs[i - 1].content if i > 0 else ""
                        st.button("📨 담당자에게 문의하기", key=f"staff_{mid}", on_click=go_inquiry, args=(prev_q,))
                    fb_key = f"fb_{mid}"
                    if fb_key not in st.session_state and meta.get("feedback") is not None:
                        st.session_state[fb_key] = meta["feedback"]
                    st.feedback("thumbs", key=fb_key, on_change=on_feedback, args=(mid,))

        # 사용자 입력 처리
        if user_input:
            history = list(msgs)
            st.chat_message("user", avatar="🙋").markdown(user_input)

            with st.chat_message("assistant", avatar="📘"):
                sources, rewritten, no_answer = [], None, False
                if st.session_state.rag_chain is None:
                    full_response = "아직 백서가 업로드되지 않았어요. 왼쪽 관리자 영역에 PDF를 올려 주세요."
                    st.markdown(full_response)
                else:
                    try:
                        search_q = user_input
                        if st.session_state.use_context:
                            search_q = standalone_question(user_input, history)
                            if search_q != user_input:
                                rewritten = search_q
                                st.caption(f"🔁 이렇게 이해했어요: {search_q}")
                        response_stream = st.session_state.rag_chain.stream({"input": search_q})
                        full_response = st.write_stream(response_stream)
                        no_answer = NO_ANSWER in full_response
                        if not no_answer:
                            sources = to_sources(st.session_state.retriever.invoke(search_q))
                            render_sources(sources)
                    except Exception as e:
                        full_response = f"답변 생성 중 오류가 발생했습니다: {e}"
                        st.error(full_response)

            msg_id = uuid.uuid4().hex[:12]
            if st.session_state.rag_chain is not None:
                msg_id = log_event("chat", user_input, full_response, no_answer,
                                   pages=[s["page"] for s in sources], extra={"rewritten": rewritten})

            # 대화 기록 저장 (기존 코드 방식 + 근거·통계 정보)
            conv = st.session_state.conversations[st.session_state.current_conv]
            if not msgs:
                conv["title"] = user_input[:20]
            msgs.append(HumanMessage(content=user_input))
            msgs.append(AIMessage(content=full_response, additional_kwargs={
                "id": msg_id, "sources": sources, "rewritten": rewritten,
                "no_answer": no_answer, "feedback": None}))
            st.rerun()


# =====================================================================
# 페이지 ② 자주 묻는 질문 (FAQ)
# =====================================================================
def page_faq():
    st.subheader("❓ 자주 묻는 질문")
    st.caption("답변은 현재 백서에서 찾아 보여 드립니다. 한 번 확인한 답변은 다른 학습자도 바로 볼 수 있어요.")
    if st.session_state.rag_chain is None:
        st.info("백서를 먼저 올려 주세요.")
        return

    with st.spinner("백서에서 FAQ를 정리하는 중..."):
        faqs = faq_questions()
    for i, q in enumerate(faqs):
        with st.expander(f"Q. {q}"):
            cache_key = (st.session_state.doc_hash, json.dumps(CFG["settings"], sort_keys=True), q)
            cached = STORE["faq_cache"].get(cache_key)
            if cached:
                st.markdown(cached["answer"])
                render_sources(cached["sources"], nested=True)
            elif st.button("답변 보기", key=f"faq_{i}"):
                with st.spinner("백서에서 찾는 중..."):
                    answer = st.session_state.rag_chain.invoke({"input": q})
                    no_answer = NO_ANSWER in answer
                    sources = [] if no_answer else to_sources(st.session_state.retriever.invoke(q))
                STORE["faq_cache"][cache_key] = {"answer": answer, "sources": sources}
                log_event("faq", q, answer, no_answer, pages=[s["page"] for s in sources])
                st.rerun()
            st.button("💬 채팅에서 이어서 질문하기", key=f"faq_chat_{i}", on_click=ask, args=(q,))

    # 다른 학습자들이 많이 물어본 질문 (질문 통계 연동)
    counter = Counter()
    original = {}
    for e in STORE["logs"]:
        if e.get("type") == "chat" and not e.get("no_answer"):
            key = normalize_q(e["question"])
            counter[key] += 1
            original.setdefault(key, e["question"])
    popular = [original[k] for k, n in counter.most_common(5) if n >= 2]
    if popular:
        st.subheader("🔥 다른 학습자들이 많이 물어본 질문")
        for i, q in enumerate(popular):
            st.button(q, key=f"pop_{i}", on_click=ask, args=(q,))


# =====================================================================
# 페이지 ③ 백서 기반 학습 계획
# =====================================================================
def page_plan():
    st.subheader("🗓️ 백서 기반 학습 계획")
    st.caption("백서에 정리된 선배들의 학습 전략과 학습 계획만 모아 내 조건에 맞게 정리합니다. "
               "백서에 없는 내용은 만들지 않고 '백서에 해당 내용 없음'으로 표시합니다.")
    if st.session_state.rag_chain is None:
        st.info("백서를 먼저 올려 주세요.")
        return

    with st.form("plan_form"):
        c1, c2 = st.columns(2)
        background = c1.radio("나의 배경", ["비전공자", "전공자", "상관없음"], horizontal=True)
        period = c2.radio("계획 단위", ["하루", "일주일", "교육 기간 전체"], horizontal=True)
        focus = st.multiselect("집중하고 싶은 부분 (선택)",
                               ["기초 다지기", "복습·학습 습관", "미니 프로젝트", "빅 프로젝트",
                                "자격증", "취업 준비", "시간 관리"])
        extra = st.text_input("추가로 고려할 점 (선택)", placeholder="예: 평일 저녁에만 복습 가능")
        submitted = st.form_submit_button("학습 계획 정리하기", type="primary")

    if submitted:
        request = (f"나의 배경: {background}\n계획 단위: {period}\n"
                   f"집중하고 싶은 부분: {', '.join(focus) or '전체'}\n추가 고려 사항: {extra or '없음'}")
        queries = ["학습 계획", "학습 전략", "공부 시간", f"{background} 학습 방법",
                   f"{period} 학습 계획", "선배 조언"] + focus + ([extra] if extra else [])
        with st.spinner("백서에서 학습 전략을 모으는 중..."):
            docs = multi_retrieve(queries, limit=10)
        text = st.write_stream(doc_task_stream(PLAN_TASK, request, docs))
        no_answer = NO_ANSWER in text
        st.session_state.plan = {"text": text, "request": request,
                                 "sources": [] if no_answer else to_sources(docs)}
        log_event("plan", request, text, no_answer)
    elif st.session_state.plan:
        st.markdown(st.session_state.plan["text"])

    plan = st.session_state.plan
    if plan:
        render_sources(plan["sources"])
        st.download_button("💾 학습 계획 저장 (Markdown)",
                           f"# 나의 학습 계획\n\n{plan['request']}\n\n---\n\n{plan['text']}",
                           file_name="aivle_study_plan.md", mime="text/markdown")


# =====================================================================
# 페이지 ④ 이해도 퀴즈 (백서에서만 출제)
# =====================================================================
def generate_quiz(topic, n):
    if topic:
        docs = multi_retrieve([topic], limit=6)
    else:
        chunks = st.session_state.doc_chunks
        docs = random.sample(chunks, min(6, len(chunks)))
    prompt = ChatPromptTemplate.from_messages([("system", QUIZ_SYSTEM), ("human", "주제: {topic}")])
    chain = prompt | get_llm().with_structured_output(Quiz)
    quiz = chain.invoke({"n": n, "context": format_docs_with_pages(docs), "topic": topic or "백서 전체"})
    items = [it.model_dump() for it in quiz.items
             if len(it.options) >= 2 and 0 <= it.answer < len(it.options)]
    return items[:n]


def page_quiz():
    st.subheader("📝 이해도 퀴즈")
    st.caption("백서 내용만으로 문제를 만듭니다. 문제마다 근거 페이지를 함께 보여 드려요.")
    if st.session_state.rag_chain is None:
        st.info("백서를 먼저 올려 주세요.")
        return

    c1, c2, c3 = st.columns([3, 1, 1])
    topic = c1.text_input("주제 (비우면 백서 전체에서 무작위 출제)", placeholder="예: 프로젝트 준비, 학습 전략")
    n = c2.selectbox("문항 수", [3, 5], index=0)
    c3.write("")
    if c3.button("문제 만들기", type="primary", width="stretch"):
        try:
            with st.spinner("백서에서 문제를 만드는 중..."):
                items = generate_quiz(topic.strip(), n)
            if items:
                st.session_state.quiz = {"id": uuid.uuid4().hex[:6], "topic": topic.strip(),
                                         "items": items, "submitted": False}
            else:
                st.warning("백서에서 문제를 만들 만한 내용을 찾지 못했어요. 다른 주제로 시도해 보세요.")
        except Exception as e:
            st.error(f"문제 생성 중 오류가 발생했습니다: {e}")

    quiz = st.session_state.quiz
    if not quiz:
        return

    with st.form(f"quiz_form_{quiz['id']}"):
        for i, item in enumerate(quiz["items"]):
            st.markdown(f"**Q{i + 1}. {item['question']}**")
            st.radio("답 선택", item["options"], index=None, key=f"qa_{quiz['id']}_{i}",
                     label_visibility="collapsed")
        graded = st.form_submit_button("채점하기", type="primary")

    if graded:
        quiz["submitted"] = True
        quiz["answers"] = [st.session_state.get(f"qa_{quiz['id']}_{i}") for i in range(len(quiz["items"]))]
        score = sum(1 for mine, it in zip(quiz["answers"], quiz["items"])
                    if mine == it["options"][it["answer"]])
        quiz["score"] = score
        log_event("quiz", quiz["topic"] or "백서 전체", f"{score}/{len(quiz['items'])}",
                  extra={"score": score, "total": len(quiz["items"])})

    if quiz.get("submitted"):
        total = len(quiz["items"])
        st.metric("내 점수", f"{quiz['score']} / {total}")
        if quiz["score"] == total:
            st.balloons()
        for i, it in enumerate(quiz["items"]):
            mine = quiz["answers"][i]
            correct = it["options"][it["answer"]]
            mark = "✅" if mine == correct else "❌"
            with st.container(border=True):
                st.markdown(f"{mark} **Q{i + 1}. {it['question']}**")
                st.markdown(f"- 내 답: {mine or '선택 안 함'}\n- 정답: **{correct}**")
                st.caption(f"해설: {it['explanation']} (근거 {it['page']})")


# =====================================================================
# 페이지 ⑤ 담당자 문의
# =====================================================================
def page_inquiry():
    st.subheader("📨 담당자 문의")
    st.caption("백서에서 답을 찾지 못한 질문은 운영 담당자에게 남겨 주세요. "
               "답변은 아래 '내 문의 내역'에서 확인할 수 있습니다.")

    if st.session_state.inquiry_prefill:
        st.session_state.inq_content = (f"[학습 도우미에서 답을 찾지 못한 질문]\n"
                                        f"{st.session_state.inquiry_prefill}\n\n")
        st.session_state.inquiry_prefill = None

    with st.form("inquiry_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("이름 *")
        contact = c2.text_input("이메일 또는 연락처 *")
        category = st.selectbox("문의 분류", INQUIRY_CATEGORIES)
        content = st.text_area("문의 내용 *", key="inq_content", height=160)
        agree = st.checkbox("문의 답변을 위한 개인정보(이름, 연락처) 수집·이용에 동의합니다. *")
        sent = st.form_submit_button("문의 보내기", type="primary")

    if sent:
        if not (name.strip() and contact.strip() and content.strip()):
            st.error("이름, 연락처, 문의 내용을 모두 입력해 주세요.")
        elif not agree:
            st.error("개인정보 수집·이용에 동의해 주세요.")
        else:
            with STORE["lock"]:
                STORE["inquiries"].append({
                    "id": uuid.uuid4().hex[:8], "time": now_str(), "session": st.session_state.session_id,
                    "name": name.strip(), "contact": contact.strip(), "category": category,
                    "content": content.strip(), "status": "접수", "reply": "", "replied_at": None,
                })
                persist("inquiries")
            st.success("문의가 접수되었습니다. 담당자가 확인 후 답변드릴게요.")

    st.subheader("📋 내 문의 내역")
    with st.expander("이전에 남긴 문의 찾기 (이름과 연락처로 조회)"):
        c1, c2 = st.columns(2)
        q_name = c1.text_input("이름", key="lookup_name")
        q_contact = c2.text_input("연락처", key="lookup_contact")
    mine = [x for x in STORE["inquiries"]
            if x["session"] == st.session_state.session_id
            or (q_name.strip() and q_contact.strip()
                and x["name"] == q_name.strip() and x["contact"] == q_contact.strip())]
    if not mine:
        st.caption("아직 남긴 문의가 없습니다.")
    for x in reversed(mine):
        badge = "badge-ok" if x["status"] == "답변 완료" else "badge-wait"
        with st.container(border=True):
            st.markdown(f"<span class='badge {badge}'>{x['status']}</span> "
                        f"<span class='badge'>{x['category']}</span> · {x['time']}", unsafe_allow_html=True)
            st.markdown(x["content"])
            if x["reply"]:
                st.info(f"💬 담당자 답변 ({x['replied_at']})\n\n{x['reply']}")


# =====================================================================
# 페이지 ⑥ 관리자 전용 페이지 (대시보드 / 문서 / 문의 / 질문 / 로그)
# =====================================================================
def admin_dashboard():
    logs = list(STORE["logs"])
    if not logs:
        st.info("아직 이용 기록이 없습니다.")
        return
    df = pd.DataFrame(logs)
    df["time"] = pd.to_datetime(df["time"])
    chat = df[df["type"] == "chat"]
    fb = df["feedback"].dropna()
    open_inq = sum(1 for x in STORE["inquiries"] if x["status"] != "답변 완료")

    m = st.columns(5)
    m[0].metric("채팅 질문", len(chat))
    m[1].metric("근거 없음 비율", f"{(chat['no_answer'].mean() * 100 if len(chat) else 0):.0f}%")
    m[2].metric("👍 / 👎", f"{int((fb == 1).sum())} / {int((fb == 0).sum())}")
    m[3].metric("미답변 문의", open_inq)
    m[4].metric("이용자 수", df["session"].nunique())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**날짜별 이용**")
        st.bar_chart(df.groupby([df["time"].dt.date, "type"]).size().unstack(fill_value=0))
    with c2:
        st.markdown("**시간대별 질문**")
        st.bar_chart(chat.groupby(chat["time"].dt.hour).size().reindex(range(24), fill_value=0))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**많이 물어본 질문 TOP 10**")
        counter, original = Counter(), {}
        for q in chat["question"]:
            key = normalize_q(q)
            counter[key] += 1
            original.setdefault(key, q)
        st.dataframe(pd.DataFrame([{"질문": original[k], "횟수": n} for k, n in counter.most_common(10)]),
                     hide_index=True, width="stretch")
    with c2:
        st.markdown("**질문 키워드 TOP 15**")
        words = Counter(w for q in chat["question"] for w in re.findall(r"[가-힣A-Za-z]{2,}", q)
                        if w not in STOPWORDS)
        if words:
            st.bar_chart(pd.Series(dict(words.most_common(15))).sort_values())

    st.markdown("**📌 백서 보완이 필요한 질문** (근거를 찾지 못한 질문 → 백서 업데이트에 활용)")
    gaps = chat[chat["no_answer"]].sort_values("time", ascending=False)
    st.dataframe(gaps[["time", "question"]].head(30).rename(columns={"time": "시각", "question": "질문"}),
                 hide_index=True, width="stretch")

    st.markdown("**👎 받은 답변**")
    bad = df[df["feedback"] == 0].sort_values("time", ascending=False)
    st.dataframe(bad[["time", "question", "answer"]].head(30)
                 .rename(columns={"time": "시각", "question": "질문", "answer": "답변"}),
                 hide_index=True, width="stretch")

    st.download_button("📥 전체 로그 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="aivle_logs.csv", mime="text/csv")


def admin_documents():
    st.markdown("**문서 등록** · 등록한 문서는 모든 사용자가 선택할 수 있고, 기본 문서는 접속하자마자 자동으로 준비됩니다.")
    files = st.file_uploader("PDF 등록 (여러 개 가능)", type=["pdf"], accept_multiple_files=True,
                             key=f"admin_up_{st.session_state.admin_upload_key}")
    if files and st.button("라이브러리에 등록", type="primary"):
        for f in files:
            add_to_library(f.getvalue(), f.name)
        st.session_state.admin_upload_key += 1
        st.toast(f"{len(files)}개 문서를 등록했습니다.")
        st.rerun()

    if not STORE["docs"]:
        st.caption("등록된 문서가 없습니다.")
    for h, meta in list(STORE["docs"].items()):
        is_default = CFG.get("default_doc") == h
        is_current = st.session_state.doc_hash == h
        with st.container(border=True):
            tags = ("⭐ 기본 문서 " if is_default else "") + ("🟢 사용 중" if is_current else "")
            st.markdown(f"**{meta['name']}** {tags}")
            st.caption(f"{meta['size'] / 1024:.0f}KB · 페이지 {meta.get('pages') or '-'} · "
                       f"조각 {meta.get('chunks') or '-'} · 등록 {meta['uploaded_at']}")
            b = st.columns(4)
            if b[0].button("기본 문서로 지정", key=f"def_{h}", disabled=is_default, width="stretch"):
                with STORE["lock"]:
                    CFG["default_doc"] = h
                    persist("config")
                st.rerun()
            if b[1].button("지금 사용", key=f"use_{h}", disabled=is_current, width="stretch"):
                load_library_doc(h)
                st.rerun()
            b[2].download_button("다운로드", Path(meta["path"]).read_bytes(), file_name=meta["name"],
                                 mime="application/pdf", key=f"dl_{h}", width="stretch")
            if b[3].button("삭제", key=f"del_{h}", disabled=meta.get("bundled", False), width="stretch"):
                delete_library_doc(h)
                st.rerun()

    st.divider()
    st.markdown("**RAG 설정** · 바꾸면 모든 사용자의 문서가 새 설정으로 다시 준비됩니다.")
    s = CFG["settings"]
    c1, c2, c3 = st.columns(3)
    k = c1.slider("검색 문서 수 (k)", 1, 10, s["k"])
    size = c2.number_input("chunk_size", 200, 3000, s["chunk_size"], step=100)
    overlap = c3.number_input("chunk_overlap", 0, 1000, s["chunk_overlap"], step=50)
    if st.button("설정 저장 · 재벡터화"):
        with STORE["lock"]:
            CFG["settings"] = {"k": int(k), "chunk_size": int(size), "chunk_overlap": int(overlap)}
            STORE["faq_cache"].clear()
            persist("config")
        st.rerun()

    st.markdown("**🔎 검색 테스트** · 질문을 넣으면 어떤 조각이 검색되는지 보여줍니다.")
    test_q = st.text_input("테스트 질문")
    if test_q and st.session_state.retriever is not None:
        for d in st.session_state.retriever.invoke(test_q):
            with st.container(border=True):
                st.caption(page_label(d))
                st.text(d.page_content[:400])


def admin_inquiries():
    status_filter = st.radio("상태", ["전체", "접수", "답변 완료"], horizontal=True)
    items = [x for x in reversed(STORE["inquiries"]) if status_filter == "전체" or x["status"] == status_filter]
    if not items:
        st.caption("문의가 없습니다.")
    for x in items:
        with st.expander(f"[{x['status']}] {x['category']} · {x['name']} · {x['time']}"):
            st.markdown(f"**연락처:** {x['contact']}")
            st.markdown(x["content"])
            reply = st.text_area("답변", value=x["reply"], key=f"reply_{x['id']}", height=120)
            if st.button("답변 저장", key=f"save_{x['id']}"):
                with STORE["lock"]:
                    x["reply"] = reply.strip()
                    x["status"] = "답변 완료" if reply.strip() else "접수"
                    x["replied_at"] = now_str() if reply.strip() else None
                    persist("inquiries")
                st.toast("답변을 저장했습니다.")
                st.rerun()


def admin_questions():
    source = st.radio("추천 질문·FAQ 방식", ["auto", "manual"], horizontal=True,
                      index=0 if CFG.get("question_source") == "auto" else 1,
                      format_func=lambda v: "🤖 백서로 자동 생성" if v == "auto" else "✍️ 직접 입력한 목록")
    if source != CFG.get("question_source"):
        with STORE["lock"]:
            CFG["question_source"] = source
            persist("config")
        st.rerun()
    if source == "auto":
        st.caption("현재 업로드된 백서를 읽고 '백서에 답이 있는 질문'만 자동으로 만듭니다. "
                   "FAQ는 백서의 FAQ 부분에서 뽑습니다. 생성에 실패하면 아래 직접 입력한 목록을 씁니다.")
        if st.session_state.rag_chain is not None:
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**자동 생성된 추천 질문**")
                for q in recommended_questions():
                    st.markdown(f"- {q}")
            with p2:
                st.markdown("**백서에서 뽑은 FAQ**")
                for q in faq_questions():
                    st.markdown(f"- {q}")
            if st.button("🔄 백서로 다시 만들기 (백서를 수정했을 때)"):
                suggest_questions.clear()
                st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        q_text = st.text_area("💡 추천 질문 직접 입력 (한 줄에 하나)", "\n".join(CFG["questions"]), height=220)
    with c2:
        f_text = st.text_area("❓ FAQ 질문 직접 입력 (한 줄에 하나)", "\n".join(CFG["faq"]), height=220)
    if st.button("저장", type="primary"):
        with STORE["lock"]:
            CFG["questions"] = [q.strip() for q in q_text.splitlines() if q.strip()]
            CFG["faq"] = [q.strip() for q in f_text.splitlines() if q.strip()]
            persist("config")
        st.toast("저장했습니다.")
        st.rerun()
    if st.button("FAQ 답변 캐시 비우기 (백서를 수정했을 때)"):
        STORE["faq_cache"].clear()
        st.toast("FAQ 답변을 다시 만들도록 초기화했습니다.")


def admin_logs():
    logs = list(reversed(STORE["logs"]))
    st.caption(f"최근 기록 {len(logs)}건")
    if logs:
        st.dataframe(pd.DataFrame(logs)[["time", "type", "question", "no_answer", "feedback", "doc"]].head(300),
                     hide_index=True, width="stretch")
        st.download_button("📥 로그 JSON", json.dumps(STORE["logs"], ensure_ascii=False, indent=1),
                           file_name="aivle_logs.json", mime="application/json")
    confirm = st.checkbox("이용 기록을 모두 삭제합니다 (되돌릴 수 없음)")
    if st.button("로그 초기화", disabled=not confirm):
        with STORE["lock"]:
            STORE["logs"].clear()
            persist("logs")
        st.rerun()


def page_admin():
    if not st.session_state.is_admin:
        st.warning("관리자만 볼 수 있는 페이지입니다. 사이드바에서 로그인해 주세요.")
        return
    st.subheader("🛠️ 관리자")
    tabs = st.tabs(["📊 질문 통계", "📚 문서 관리", "📨 문의 관리", "💡 추천 질문·FAQ", "🗂️ 로그"])
    with tabs[0]:
        admin_dashboard()
    with tabs[1]:
        admin_documents()
    with tabs[2]:
        admin_inquiries()
    with tabs[3]:
        admin_questions()
    with tabs[4]:
        admin_logs()


# =====================================================================
# 본문: 헤더 박스 (교안 44쪽) → 페이지 메뉴 → 선택한 페이지
# =====================================================================
st.markdown(f"""<div class="header-box"><img src="{LOGO_URL}" width="44">
    <div><h2>AIVLE School 학습 도우미</h2>
    <p>선배들의 경험과 학습 전략을 담은 AIVLE School 백서를 근거로 답변합니다.</p></div></div>""",
            unsafe_allow_html=True)

PAGES = {
    "chat": st.Page(page_chat, title="학습 도우미", icon="💬", default=True),
    "faq": st.Page(page_faq, title="FAQ", icon="❓"),
    "plan": st.Page(page_plan, title="학습 계획", icon="🗓️"),
    "quiz": st.Page(page_quiz, title="이해도 퀴즈", icon="📝"),
    "inquiry": st.Page(page_inquiry, title="담당자 문의", icon="📨"),
}
if st.session_state.is_admin:
    PAGES["admin"] = st.Page(page_admin, title="관리자", icon="🛠️")

current_page = st.navigation(list(PAGES.values()), position="top")

# 다른 페이지에서 요청한 이동 (추천 질문 클릭 → 채팅, 근거 없음 → 담당자 문의 등)
target = st.session_state.goto
if target:
    st.session_state.goto = None
    if target in PAGES and PAGES[target].url_path != current_page.url_path:
        st.switch_page(PAGES[target])

current_page.run()
