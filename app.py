"""
AIVLE School 학습 도우미 (DX 트랙 미니 프로젝트 4차)

[기능 명세]
■ 필수 기능
 F1. Streamlit Community Cloud 기반 인터페이스 (교안 44쪽 배치)
     - 사이드바: 로고 카드 → 관리자(파일 업로드) → 추천 질문 → 대화 삭제 → 관리자 설정
     - 본문: 헤더 박스 → 점선 채팅 박스(대화 내용 + 입력창)
     - OpenAI API Key: st.secrets → 환경변수 → 사이드바 직접 입력 순으로 사용
 F2. 백서 PDF 직접 업로드
     - F2-1 드래그 앤 드롭: 사이드바 '관리자' 영역 업로더에 파일을 끌어다 놓기
     - F2-2 '+' 버튼: 채팅 입력창의 + 버튼으로 PDF 첨부 (질문과 함께 첨부 가능)
 F3. 업로드 즉시 자동 벡터화 → RAG 체인 구성 (rag_pipeline 함수 사용)
     - 업로드 파일을 임시 .pdf 파일로 저장한 뒤 "파일 경로"를 rag_pipeline에 전달
     - 만든 체인은 st.session_state에 저장하고 파일 해시로 비교 → 재실행돼도 다시 벡터화하지 않음
     - 사용자(세션)마다 별도 Chroma 컬렉션 사용 → 여러 명이 동시에 써도 문서가 섞이지 않음
 F4. 백서 기반의 명확한 응답
     - 시스템 메시지에 "참고 문서만 근거로" 규칙 명시, temperature=0
     - 근거가 없으면 정해진 문장("백서에서 해당 내용을 확인할 수 없습니다.")만 답변
     - 문서가 없을 때는 LLM을 호출하지 않고 업로드 안내만 표시
     - 답변 아래에 근거가 된 백서 페이지·발췌문 표시 (근거 없음 답변에는 표시하지 않음)

■ 옵션 기능
 O1. 관리자 설정 (사이드바 하단, 비밀번호: st.secrets["ADMIN_PASSWORD"], 없으면 기본값)
     - 문서 정보(파일명, 페이지 수, 청크 수, 처리 시각)
     - RAG 설정 변경(k, chunk_size, chunk_overlap) 후 재벡터화
     - 검색 테스트(질문 → 어떤 청크가 검색되는지 확인)
     - 추천 질문 편집
     - 이용 통계(질문 수, 근거 없음 답변 수, 👍/👎 피드백)
     - 대화 로그 JSON 다운로드
 O2. 추천 질문: 첫 화면 카드 + 사이드바 버튼, 클릭 시 바로 질문
 O3. 대화 삭제: 사이드바 버튼으로 대화·피드백 초기화
 O4. 문서 변경: 새 PDF 업로드 시 자동 교체, '문서 해제' 버튼
 O5. 기타: 답변별 👍/👎 피드백, 대화 내용 Markdown 다운로드, 처리 진행 상태 표시
"""

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime
from operator import itemgetter

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------
# 0. 기본 설정 (set_page_config는 반드시 첫 Streamlit 호출이어야 함)
# ---------------------------------------------------------------------
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="📘", layout="wide")

LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
NO_ANSWER = "백서에서 해당 내용을 확인할 수 없습니다."

DEFAULT_SETTINGS = {"k": 4, "chunk_size": 1000, "chunk_overlap": 200}

DEFAULT_QUESTIONS = [
    "AIVLE School은 어떤 교육 과정인가요?",
    "DX 트랙에서는 무엇을 배우나요?",
    "비전공자는 무엇부터 공부해야 하나요?",
    "하루 공부 시간은 어느 정도가 적당한가요?",
    "프로젝트는 어떻게 준비해야 하나요?",
    "선배들이 꼭 해주고 싶은 조언은 무엇인가요?",
    "취업을 잘하려면 무엇이 중요한가요?",
    "자주 묻는 질문(FAQ)을 알려주세요.",
]

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


# ---------------------------------------------------------------------
# 1. RAG 파이프라인 (미션 2 표준안 + 앱에서 쓰기 위한 매개변수 추가)
# ---------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_pipeline(
    filepath,
    chunk_size=1000,
    chunk_overlap=200,
    k=4,
    collection_name="aivle_docs",
    return_details=False,
):
    """PDF 경로를 받아 RAG 체인을 반환한다.
    return_details=True이면 (체인, 상세정보) 튜플을 반환한다(근거 표시·관리자 화면용)."""
    # 1. PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(str(filepath))
    docs = loader.load()

    # 비어 있는 문서 제외
    docs = [doc for doc in docs if doc.page_content.strip()]
    if not docs:
        raise ValueError("PDF에서 읽을 수 있는 본문이 없습니다.")
    print(f"1. PDF 읽기 완료: {len(docs)}개 Document")

    # 2. 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"2. 문서 분할 완료: {len(chunks)}개 조각")

    # 3. 임베딩 모델
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. 벡터 저장소 구성 (재실행 시 중복 저장 방지를 위해 초기화)
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings)
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)
    print("3. 임베딩 및 벡터 저장 완료")

    # 5. 검색기 구성
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 6. 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])

    # 7. 답변 모델
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    # 8. RAG 체인 연결
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("4. RAG 체인 구성 완료")

    if return_details:
        return rag_chain, {"retriever": retriever, "pages": len(docs), "chunks": len(chunks)}
    return rag_chain


# ---------------------------------------------------------------------
# 2. 세션 상태 초기화
# ---------------------------------------------------------------------
def init_state():
    defaults = {
        "session_id": uuid.uuid4().hex[:8],
        "messages": [],              # {"role", "content", "sources", "time"}
        "rag_chain": None,
        "retriever": None,
        "doc_hash": None,
        "doc_bytes": None,
        "doc_info": None,
        "settings": dict(DEFAULT_SETTINGS),
        "questions": list(DEFAULT_QUESTIONS),
        "pending_question": None,
        "uploader_key": 0,
        "last_sidebar_file_id": None,
        "is_admin": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ---------------------------------------------------------------------
# 3. 보조 함수
# ---------------------------------------------------------------------
def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:  # secrets.toml이 없는 로컬 환경
        return default


def ensure_api_key():
    key = get_secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or st.session_state.get("user_api_key")
    if key:
        os.environ["OPENAI_API_KEY"] = key
    return key


def page_label(doc):
    page = doc.metadata.get("page")
    # PyMuPDF 계열 로더는 페이지 번호를 0부터 매긴다 → 사람이 보는 번호로 +1
    return f"p.{page + 1}" if isinstance(page, int) else "페이지 정보 없음"


def build_chain(file_bytes, file_name, force=False):
    """업로드 파일 → 임시 파일 경로 → rag_pipeline. 같은 파일이면 다시 만들지 않는다."""
    file_hash = hashlib.md5(file_bytes).hexdigest()
    if not force and st.session_state.doc_hash == file_hash:
        return

    if not ensure_api_key():
        st.error("OpenAI API Key가 없어 문서를 처리할 수 없습니다. 사이드바에서 입력해 주세요.")
        return

    cfg = st.session_state.settings
    with st.status(f"📄 '{file_name}' 분석 중...", expanded=True) as status:
        # ① 임시 파일 생성 ② 업로드 내용 저장 ③ 경로 확보 ④ 파이프라인에 경로 전달
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            st.write("문서 읽기 → 분할 → 임베딩 → 체인 구성")
            chain, details = rag_pipeline(
                tmp_path,
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                k=cfg["k"],
                collection_name=f"aivle_{st.session_state.session_id}",
                return_details=True,
            )
        except Exception as e:
            status.update(label=f"❌ 문서 처리 실패: {e}", state="error")
            return
        finally:
            os.remove(tmp_path)

        st.session_state.rag_chain = chain
        st.session_state.retriever = details["retriever"]
        st.session_state.doc_hash = file_hash
        st.session_state.doc_bytes = file_bytes
        st.session_state.doc_info = {
            "name": file_name,
            "pages": details["pages"],
            "chunks": details["chunks"],
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **cfg,
        }
        status.update(label=f"✅ '{file_name}' 준비 완료 ({details['chunks']}개 조각)", state="complete", expanded=False)


def unload_document():
    for key in ["rag_chain", "retriever", "doc_hash", "doc_bytes", "doc_info", "last_sidebar_file_id"]:
        st.session_state[key] = None
    st.session_state.uploader_key += 1  # 업로더 위젯 비우기


def clear_chat():
    st.session_state.messages = []
    for key in [k for k in st.session_state.keys() if str(k).startswith("fb_")]:
        del st.session_state[key]


def ask(question):
    st.session_state.pending_question = question


def chat_as_markdown():
    lines = ["# AIVLE School 학습 도우미 대화 기록", ""]
    for m in st.session_state.messages:
        who = "🙋 질문" if m["role"] == "user" else "📘 답변"
        lines += [f"### {who} ({m['time']})", m["content"], ""]
    return "\n".join(lines)


def feedback_stats():
    values = [st.session_state.get(f"fb_{i}") for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"]
    return values.count(1), values.count(0)


# ---------------------------------------------------------------------
# 4. 스타일
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}

    /* 사이드바 로고 카드 (교안: 로고 + 소개 문구 박스) */
    .logo-card {border: 1px solid #d0d5da; border-radius: 12px; padding: .9rem; background: #f8f9fa;
                text-align: center; margin-bottom: .8rem;}
    .logo-card img {width: 130px;}
    .logo-card p {font-size: .8rem; color: #6c757d; margin: .4rem 0 0 0;}

    /* 헤더 박스 (교안: 로고 아이콘 + 제목 + 부제가 둥근 박스 안에) */
    .header-box {border: 1px solid #d0d5da; border-radius: 12px; padding: .9rem 1.2rem; background: #f8f9fa;
                 display: flex; align-items: center; gap: .8rem; margin-bottom: 1rem;}
    .header-box h2 {margin: 0; padding: 0; font-size: 1.6rem;}
    .header-box p {margin: .1rem 0 0 0; color: #6c757d; font-size: .9rem;}

    /* 채팅 영역 (교안: 점선 테두리 박스, 입력창도 박스 안) */
    .st-key-chat_area {border: 2px dashed #3aa8a0; border-radius: 12px; padding: 1rem;}

    .hero {background: #3aa8a00d; border: 1px solid #3aa8a040; border-radius: 12px;
           padding: 1rem 1.2rem; margin-bottom: .8rem;}
    .hero h4 {margin: 0 0 .3rem 0;}
    .doc-badge {display:inline-block; background:#f1f3f5; border-radius:999px; padding:.2rem .7rem; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# 5. 사이드바
# ---------------------------------------------------------------------
with st.sidebar:
    # --- 로고 영역 ---
    st.markdown(
        f"""<div class="logo-card"><img src="{LOGO_URL}">
        <p>에이블스쿨 백서를 바탕으로<br>학습 고민에 답해 드립니다.</p></div>""",
        unsafe_allow_html=True,
    )

    # API Key (secrets에 없을 때만 입력받음)
    if not (get_secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        st.text_input("🔑 OpenAI API Key", type="password", key="user_api_key")

    # --- 관리자: 파일 업로드 (교안 배치, 누구나 업로드 가능) ---
    # F2-1 / O4: 문서 업로드(드래그 앤 드롭) 및 변경
    st.subheader("관리자")
    uploaded = st.file_uploader(
        "PDF를 여기로 드래그하거나 클릭해 선택하세요",
        type=["pdf"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    # 업로더는 파일을 계속 들고 있으므로, '새로 올린 경우'에만 처리
    if uploaded is not None and uploaded.file_id != st.session_state.last_sidebar_file_id:
        st.session_state.last_sidebar_file_id = uploaded.file_id
        build_chain(uploaded.getvalue(), uploaded.name)

    info = st.session_state.doc_info
    if info:
        st.success(f"사용 중: **{info['name']}**\n\n{info['pages']}페이지 · {info['chunks']}개 조각")
        st.button("📤 문서 해제", on_click=unload_document, use_container_width=True)
    else:
        st.info("백서 PDF를 올리면 바로 질문할 수 있어요.\n\n채팅창의 **+** 버튼으로도 첨부할 수 있습니다.")

    # --- O2: 추천 질문 (문서가 없으면 비활성) ---
    st.subheader("💡 추천 질문")
    for i, q in enumerate(st.session_state.questions):
        st.button(q, key=f"side_q_{i}", on_click=ask, args=(q,), use_container_width=True,
                  disabled=st.session_state.rag_chain is None)

    # --- O3: 대화 삭제 / O5: 대화 저장 ---
    st.button("🗑️ 대화 삭제", on_click=clear_chat, use_container_width=True,
              disabled=not st.session_state.messages)
    st.download_button("💾 대화 저장 (Markdown)", chat_as_markdown(), file_name="aivle_chat.md",
                       mime="text/markdown", use_container_width=True,
                       disabled=not st.session_state.messages)

    # --- O1: 관리자 설정 (비밀번호) ---
    st.divider()
    with st.expander("🔐 관리자 설정", expanded=st.session_state.is_admin):
        if not st.session_state.is_admin:
            pw = st.text_input("비밀번호", type="password", key="admin_pw")
            if st.button("로그인", use_container_width=True):
                if pw == get_secret("ADMIN_PASSWORD", "aivle"):
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
        else:
            tab_doc, tab_set, tab_q, tab_stat = st.tabs(["문서", "설정", "질문", "통계"])

            with tab_doc:
                if info:
                    st.json(info)
                    test_q = st.text_input("🔎 검색 테스트", placeholder="질문을 넣으면 검색 결과를 보여줘요")
                    if test_q:
                        for d in st.session_state.retriever.invoke(test_q):
                            st.caption(page_label(d))
                            st.text(d.page_content[:300])
                else:
                    st.caption("업로드된 문서가 없습니다.")

            with tab_set:
                s = st.session_state.settings
                k = st.slider("검색 문서 수 (k)", 1, 10, s["k"])
                size = st.number_input("chunk_size", 200, 3000, s["chunk_size"], step=100)
                overlap = st.number_input("chunk_overlap", 0, 1000, s["chunk_overlap"], step=50)
                if st.button("설정 적용 · 재벡터화", use_container_width=True,
                             disabled=st.session_state.doc_bytes is None):
                    st.session_state.settings = {"k": k, "chunk_size": int(size), "chunk_overlap": int(overlap)}
                    build_chain(st.session_state.doc_bytes, info["name"], force=True)

            with tab_q:
                edited = st.text_area("추천 질문 (한 줄에 하나)", "\n".join(st.session_state.questions), height=200)
                if st.button("추천 질문 저장", use_container_width=True):
                    st.session_state.questions = [q.strip() for q in edited.splitlines() if q.strip()]
                    st.rerun()

            with tab_stat:
                msgs = st.session_state.messages
                answers = [m for m in msgs if m["role"] == "assistant"]
                up, down = feedback_stats()
                c1, c2 = st.columns(2)
                c1.metric("질문 수", len(answers))
                c2.metric("근거 없음", sum(NO_ANSWER in m["content"] for m in answers))
                c1.metric("👍", up)
                c2.metric("👎", down)
                st.download_button("대화 로그(JSON)", json.dumps(msgs, ensure_ascii=False, indent=2),
                                   file_name="chat_log.json", mime="application/json", use_container_width=True)

            if st.button("로그아웃", use_container_width=True):
                st.session_state.is_admin = False
                st.rerun()


# ---------------------------------------------------------------------
# 6. 메인 화면: 헤더 박스 (교안 배치)
# ---------------------------------------------------------------------
st.markdown(
    f"""<div class="header-box">
    <img src="{LOGO_URL}" width="44">
    <div><h2>AIVLE School 학습 도우미</h2>
    <p>선배들의 경험과 학습 전략을 담은 AIVLE School 백서를 근거로 답변합니다.</p></div>
    </div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 7. 채팅 영역: 점선 박스 안에 [대화 내용 → 입력창] (교안 배치)
# ---------------------------------------------------------------------
with st.container(key="chat_area"):
    msg_box = st.container(height=520, border=False)   # 대화가 길어지면 이 안에서 스크롤
    # F2-2: '+' 버튼으로 PDF 첨부 가능한 채팅 입력창 (Streamlit 1.43+)
    chat_value = st.chat_input(
        "질문을 입력하세요  (+ 버튼으로 백서 PDF 첨부)",
        accept_file=True,
        file_type=["pdf"],
    )

question = None
if chat_value:
    if chat_value.files:
        f = chat_value.files[0]
        with msg_box:
            build_chain(f.getvalue(), f.name)
    if chat_value.text:
        question = chat_value.text.strip()
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


def render_sources(sources):
    if sources:
        with st.expander(f"📎 답변 근거 ({', '.join(dict.fromkeys(s['page'] for s in sources))})"):
            for s in sources:
                st.markdown(f"**{s['page']}**")
                st.caption(s["text"])


with msg_box:
    # 첫 화면 (대화가 없을 때)
    if not st.session_state.messages and not question:
        if st.session_state.rag_chain:
            st.markdown(
                f"""<div class="hero"><h4>무엇이 궁금하신가요? 🙌</h4>
                <span class="doc-badge">📄 {st.session_state.doc_info['name']}</span>
                &nbsp;아래 질문을 눌러 바로 시작해 보세요.</div>""",
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            for i, q in enumerate(st.session_state.questions[:6]):
                cols[i % 2].button(f"💡 {q}", key=f"main_q_{i}", on_click=ask, args=(q,), use_container_width=True)
        else:
            st.markdown(
                """<div class="hero"><h4>시작하려면 백서를 올려 주세요 📂</h4>
                왼쪽 <b>관리자</b> 영역에 PDF를 <b>드래그</b>하거나, 아래 입력창의 <b>+</b> 버튼으로 첨부하면
                자동으로 분석한 뒤 바로 질문할 수 있어요.</div>""",
                unsafe_allow_html=True,
            )

    # 기존 대화 출력
    for i, m in enumerate(st.session_state.messages):
        avatar = "🙋" if m["role"] == "user" else "📘"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("sources"))
                st.feedback("thumbs", key=f"fb_{i}")

    # -----------------------------------------------------------------
    # 8. 새 질문 처리 (F4)
    # -----------------------------------------------------------------
    if question:
        now = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({"role": "user", "content": question, "sources": [], "time": now})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="📘"):
            sources = []
            if st.session_state.rag_chain is None:
                answer = "아직 백서가 업로드되지 않았어요. 왼쪽 관리자 영역에 PDF를 드래그하거나 **+** 버튼으로 첨부해 주세요."
                st.markdown(answer)
            elif not ensure_api_key():
                answer = "OpenAI API Key가 설정되지 않았습니다."
                st.markdown(answer)
            else:
                try:
                    answer = st.write_stream(st.session_state.rag_chain.stream({"input": question}))
                    if NO_ANSWER not in answer:
                        docs = st.session_state.retriever.invoke(question)
                        sources = [{"page": page_label(d), "text": d.page_content[:300]} for d in docs]
                        render_sources(sources)
                except Exception as e:
                    answer = f"답변 생성 중 오류가 발생했습니다: {e}"
                    st.error(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources, "time": datetime.now().strftime("%H:%M")}
        )
        st.rerun()  # 피드백 버튼까지 포함해 대화 영역을 다시 그림
