
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter
import os
import json
import uuid
import hashlib
import tempfile
from datetime import datetime
import warnings
warnings.filterwarnings(action="ignore")
try:
    import pymupdf
    pymupdf.set_messages(path=os.devnull)   # PDF 읽을 때 경고 메시지 감추기 (노트북과 동일)
except Exception:
    pass

# =====================================================================
# AIVLE School 학습 도우미 (기존 배포 코드 기반 + 교안 44쪽 배치 + 46쪽 주의 사항)
#
# [필수] F1 Streamlit 인터페이스 / F2 업로드(드래그 + '+'버튼) / F3 자동 벡터화(rag_pipeline)
#        F4 백서 기반 응답(근거 없으면 정해진 문장, 근거 페이지 표시)
# [옵션] O1 관리자 설정 / O2 추천 질문 / O3 대화 삭제 / O4 문서 변경 / O5 피드백·대화 저장
# =====================================================================

APP_VERSION = "v4 · 노트북 환경 반영"   # 배포 확인용 (사이드바 맨 아래 표시)
LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
NO_ANSWER = "백서에서 해당 내용을 확인할 수 없습니다."

DEFAULT_QUESTIONS = [
    "AIVLE School은 어떤 교육 과정인가요?",
    "비전공자는 무엇부터 공부해야 하나요?",
    "하루 공부 시간은 어느 정도가 적당한가요?",
    "프로젝트는 어떻게 준비해야 하나요?",
    "선배들이 꼭 해주고 싶은 조언은 무엇인가요?",
    "취업을 잘하려면 무엇이 중요한가요?",
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


# 문서 포맷팅 함수 (기존 코드 그대로)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================================
# RAG 파이프라인 (미션 2 표준안) - 파일 "경로"를 받아 RAG 체인 반환
# =====================================================================
# 01번 노트북 함수와 같은 동작: rag_pipeline(경로)만 넣으면 기존과 똑같이 체인을 반환한다.
# 뒤의 매개변수는 관리자 설정·근거 표시용으로만 추가했고, 기본값은 01번과 동일하다.
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
        return rag_chain, {"retriever": retriever, "pages": len(docs), "chunks": len(chunks)}
    return rag_chain


# =====================================================================
# 페이지 설정
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
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 세션 상태 초기화
# =====================================================================
defaults = {
    "session_id": uuid.uuid4().hex[:8],
    "messages": [],              # HumanMessage / AIMessage (기존 코드 방식)
    "rag_chain": None,           # 교안 46쪽: RAG 체인은 세션 상태에 저장
    "retriever": None,
    "doc_hash": None,
    "doc_bytes": None,
    "doc_info": None,
    "settings": {"k": 4, "chunk_size": 1000, "chunk_overlap": 200},
    "questions": list(DEFAULT_QUESTIONS),
    "pending_question": None,
    "uploader_key": 0,
    "last_file_id": None,
    "is_admin": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================================
# 보조 함수
# =====================================================================
def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def build_chain(file_bytes, file_name, force=False):
    """교안 46쪽: 업로드 파일 → 임시 .pdf 파일 → 파일 경로를 rag_pipeline에 전달.
    같은 파일이면 다시 벡터화하지 않는다."""
    file_hash = hashlib.md5(file_bytes).hexdigest()
    if not force and st.session_state.doc_hash == file_hash:
        return

    cfg = st.session_state.settings
    with st.spinner(f"'{file_name}' 문서를 분석하고 있어요..."):
        # ① 임시 파일 생성 (.pdf) ② 업로드 내용 저장 ③ 경로 확보 (tmp.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            # ④ RAG 파이프라인에 "경로" 전달
            chain, details = rag_pipeline(
                tmp_path,
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                k=cfg["k"],
                collection_name=f"aivle_{st.session_state.session_id}",  # 사용자별 분리
                return_details=True,
            )
        except Exception as e:
            st.error(f"문서 처리 실패: {e}")
            return
        finally:
            os.remove(tmp_path)

    st.session_state.rag_chain = chain
    st.session_state.retriever = details["retriever"]
    st.session_state.doc_hash = file_hash
    st.session_state.doc_bytes = file_bytes
    st.session_state.doc_info = {
        "name": file_name, "pages": details["pages"], "chunks": details["chunks"],
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **cfg,
    }
    st.toast(f"✅ '{file_name}' 준비 완료")


def unload_document():
    for key in ["rag_chain", "retriever", "doc_hash", "doc_bytes", "doc_info", "last_file_id"]:
        st.session_state[key] = None
    st.session_state.uploader_key += 1


def clear_chat():
    st.session_state.messages = []
    for key in [k for k in st.session_state.keys() if str(k).startswith("fb_")]:
        del st.session_state[key]


def ask(question):
    st.session_state.pending_question = question


def page_label(doc):
    page = doc.metadata.get("page")
    return f"p.{page + 1}" if isinstance(page, int) else "페이지 정보 없음"


def render_sources(sources):
    if sources:
        pages = ", ".join(dict.fromkeys(s["page"] for s in sources))
        with st.expander(f"📎 답변 근거 ({pages})"):
            for s in sources:
                st.markdown(f"**{s['page']}**")
                st.caption(s["text"])


def chat_as_markdown():
    lines = ["# AIVLE School 학습 도우미 대화 기록", ""]
    for m in st.session_state.messages:
        who = "🙋 질문" if isinstance(m, HumanMessage) else "📘 답변"
        lines += [f"### {who}", m.content, ""]
    return "\n".join(lines)


# --- Streamlit 버전이 낮아도 멈추지 않도록 하는 호환 함수 ---
def chat_area_container():
    try:
        return st.container(key="chat_area")       # 점선 박스 (최신 버전)
    except TypeError:
        return st.container(border=True)           # 낮은 버전: 실선 박스


def message_container():
    try:
        return st.container(height=520, border=False)
    except TypeError:
        return st.container()


def chat_input_with_file(placeholder):
    """'+' 버튼 첨부(Streamlit 1.43+). 지원하지 않으면 일반 입력창. (text, file) 반환"""
    try:
        value = st.chat_input(placeholder, accept_file=True, file_type=["pdf"])
        if not value:
            return None, None
        files = value.files if hasattr(value, "files") else value.get("files", [])
        text = value.text if hasattr(value, "text") else value.get("text", "")
        return text, (files[0] if files else None)
    except TypeError:
        return st.chat_input(placeholder), None


# =====================================================================
# 사이드바 (교안 44쪽: 로고 → 관리자(파일 업로드) → 추천 질문 → 대화 삭제)
# =====================================================================
with st.sidebar:
    st.markdown(f"""<div class="logo-card"><img src="{LOGO_URL}">
        <p>에이블스쿨 백서를 바탕으로<br>학습 고민에 답해 드립니다.</p></div>""",
                unsafe_allow_html=True)

    # --- 관리자: 파일 업로드 (드래그 앤 드롭) ---
    st.subheader("관리자")
    uploaded = st.file_uploader("백서 PDF를 드래그하거나 클릭해 선택하세요", type=["pdf"],
                                key=f"uploader_{st.session_state.uploader_key}")
    if uploaded is not None:
        file_id = getattr(uploaded, "file_id", None) or f"{uploaded.name}_{uploaded.size}"
        if file_id != st.session_state.last_file_id:      # 새로 올린 경우에만 처리
            st.session_state.last_file_id = file_id
            build_chain(uploaded.getvalue(), uploaded.name)

    info = st.session_state.doc_info
    if info:
        st.success(f"사용 중: **{info['name']}**\n\n{info['pages']}페이지 · {info['chunks']}개 조각")
        st.button("📤 문서 해제", on_click=unload_document, width="stretch")
    else:
        st.info("백서 PDF를 올리면 바로 질문할 수 있어요.")

    # --- 추천 질문 ---
    st.subheader("💡 추천 질문")
    for i, q in enumerate(st.session_state.questions):
        st.button(q, key=f"side_q_{i}", on_click=ask, args=(q,), width="stretch",
                  disabled=st.session_state.rag_chain is None)

    # --- 대화 삭제 / 저장 ---
    st.button("🗑️ 대화 삭제", on_click=clear_chat, width="stretch",
              disabled=not st.session_state.messages)
    st.download_button("💾 대화 저장", chat_as_markdown(), file_name="aivle_chat.md",
                       mime="text/markdown", width="stretch",
                       disabled=not st.session_state.messages)

    # --- 관리자 설정 (비밀번호) ---
    st.divider()
    with st.expander("🔐 관리자 설정", expanded=st.session_state.is_admin):
        if not st.session_state.is_admin:
            pw = st.text_input("비밀번호", type="password")
            if st.button("로그인", width="stretch"):
                if pw == get_secret("ADMIN_PASSWORD", "aivle"):
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
        else:
            st.markdown("**문서 정보**")
            if info:
                st.json(info)
                test_q = st.text_input("🔎 검색 테스트")
                if test_q:
                    for d in st.session_state.retriever.invoke(test_q):
                        st.caption(page_label(d))
                        st.text(d.page_content[:300])
            else:
                st.caption("업로드된 문서가 없습니다.")

            st.markdown("**RAG 설정**")
            s = st.session_state.settings
            k = st.slider("검색 문서 수 (k)", 1, 10, s["k"])
            size = st.number_input("chunk_size", 200, 3000, s["chunk_size"], step=100)
            overlap = st.number_input("chunk_overlap", 0, 1000, s["chunk_overlap"], step=50)
            if st.button("설정 적용 · 재벡터화", width="stretch",
                         disabled=st.session_state.doc_bytes is None):
                st.session_state.settings = {"k": k, "chunk_size": int(size), "chunk_overlap": int(overlap)}
                build_chain(st.session_state.doc_bytes, info["name"], force=True)

            st.markdown("**추천 질문 편집**")
            edited = st.text_area("한 줄에 하나", "\n".join(st.session_state.questions), height=150)
            if st.button("추천 질문 저장", width="stretch"):
                st.session_state.questions = [q.strip() for q in edited.splitlines() if q.strip()]
                st.rerun()

            st.markdown("**이용 통계**")
            answers = [m for m in st.session_state.messages if isinstance(m, AIMessage)]
            fb = [st.session_state.get(f"fb_{i}") for i, m in enumerate(st.session_state.messages)
                  if isinstance(m, AIMessage)]
            st.write(f"질문 {len(answers)}개 · 근거 없음 {sum(NO_ANSWER in m.content for m in answers)}개 · "
                     f"👍 {fb.count(1)} · 👎 {fb.count(0)}")
            log = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                   for m in st.session_state.messages]
            st.download_button("대화 로그(JSON)", json.dumps(log, ensure_ascii=False, indent=2),
                               file_name="chat_log.json", mime="application/json", width="stretch")

            if st.button("로그아웃", width="stretch"):
                st.session_state.is_admin = False
                st.rerun()

    st.caption(f"앱 버전: {APP_VERSION}")


# =====================================================================
# 본문: 헤더 박스 → 점선 채팅 박스(대화 + 입력창)  (교안 44쪽)
# =====================================================================
st.markdown(f"""<div class="header-box"><img src="{LOGO_URL}" width="44">
    <div><h2>AIVLE School 학습 도우미</h2>
    <p>선배들의 경험과 학습 전략을 담은 AIVLE School 백서를 근거로 답변합니다.</p></div></div>""",
            unsafe_allow_html=True)

with chat_area_container():
    msg_box = message_container()
    user_input, attached = chat_input_with_file("질문을 입력하세요  (+ 버튼으로 백서 PDF 첨부)")

# '+' 버튼으로 첨부한 PDF 처리
if attached is not None:
    with msg_box:
        build_chain(attached.getvalue(), attached.name)

# 추천 질문 클릭
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

with msg_box:
    # 첫 화면
    if not st.session_state.messages and not user_input:
        if st.session_state.rag_chain:
            st.markdown(f"""<div class="hero"><h4>무엇이 궁금하신가요? 🙌</h4>
                📄 {st.session_state.doc_info['name']} · 아래 질문을 눌러 바로 시작해 보세요.</div>""",
                        unsafe_allow_html=True)
            cols = st.columns(2)
            for i, q in enumerate(st.session_state.questions[:6]):
                cols[i % 2].button(f"💡 {q}", key=f"main_q_{i}", on_click=ask, args=(q,),
                                   width="stretch")
        else:
            st.markdown("""<div class="hero"><h4>시작하려면 백서를 올려 주세요 📂</h4>
                왼쪽 <b>관리자</b> 영역에 PDF를 <b>드래그</b>하거나, 입력창의 <b>+</b> 버튼으로 첨부하세요.</div>""",
                        unsafe_allow_html=True)

    # 기존 대화 내용 출력 (기존 코드 방식)
    for i, message in enumerate(st.session_state.messages):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)
            if role == "assistant":
                render_sources(message.additional_kwargs.get("sources"))
                if hasattr(st, "feedback"):
                    st.feedback("thumbs", key=f"fb_{i}")

    # 사용자 입력 처리
    if user_input:
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            sources = []
            if st.session_state.rag_chain is None:
                full_response = "아직 백서가 업로드되지 않았어요. 왼쪽 관리자 영역에 PDF를 올려 주세요."
                st.markdown(full_response)
            else:
                try:
                    response_stream = st.session_state.rag_chain.stream({"input": user_input})
                    full_response = st.write_stream(response_stream)
                    if NO_ANSWER not in full_response:
                        docs = st.session_state.retriever.invoke(user_input)
                        sources = [{"page": page_label(d), "text": d.page_content[:300]} for d in docs]
                        render_sources(sources)
                except Exception as e:
                    full_response = f"답변 생성 중 오류가 발생했습니다: {e}"
                    st.error(full_response)

        # 대화 기록 저장 (기존 코드 방식 + 근거 정보)
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.messages.append(AIMessage(content=full_response,
                                                   additional_kwargs={"sources": sources}))
        st.rerun()
