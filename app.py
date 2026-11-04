
import os
import base64
import json
import requests
import io 
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="음성 AI 서비스", page_icon="🤖", layout="wide")
st.title("음성 인식 AI 서비스")
st.write("음성으로 질문하시면 답변해 드립니다.")

# 역할 지정
sys_role = """
당신은 마케팅팀의 우수 직원 '김 대리(Marketing Data Analyst)'를 밀착 보조하는 전문 AI 개인 비서 '에이블'입니다.
다음과 같은 역할을 수행합니다:
1. 마케팅 회의록을 분석하고 핵심 업무 지시사항을 정확하게 추출합니다.
2. 데이터 기반의 마케팅 전략 수립과 콘텐츠 기획을 적극적으로 지원합니다.
3. 사용자의 질문에 항상 정중하고 전문적인 어조로 친절하게 응대합니다.
"""

# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지시 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")

# 버튼을 눌렀을 때만 실행
if submit:
    # 업로드 된 파일이 있으면 디코딩
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    # 녹음이 없으면 녹음 요청
    if audio_value is None:
        st.warning("음성을 녹음해주세요.")
        st.stop()

    # 시스템 메세지
    system_content = sys_role
    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"
    input_messages = [{"role": "system", "content": system_content}]

    # STT: 음성 → 텍스트
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_value, "audio/wav"))

    # 사용자 메시지
    input_messages.append({"role": "user", "content": transcript.text})

    # LLM 호출 및 응답 듣기
    with st.spinner("답변을 준비하고 있습니다..."):
        # LLM 호출
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages
        )
        answer = response.output_text

       # [2단계 2번 수정] AI 답변 위에 음성 재생 플레이어 배치
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=answer
        )
        audio_bytes = speech_response.read()
        
        # AI 답변 영역 상단에 음성 플레이어 배치
        st.caption("🔊 AI 비서 음성 답변")
        st.audio(io.BytesIO(audio_bytes), format="audio/mp3", autoplay=False)
        
        st.subheader("AI 답변")
        st.write(answer)
        # =========================================================
# [2단계 3번] 전문 AI 개인비서 UI/UX 구성 (사이드바 확장 및 퀵 프롬프트 칩)
# =========================================================

# 1. 사이드바 확장 (비서 프로필 + 데이터 연동 상태 카드)
with st.sidebar:
    st.markdown("### 🤖 비서 프로필")
    st.info(
        "**이름:** 에이블 (AIVLE)\n"
        "**역할:** 마케팅팀 김 대리 밀착 보조 비서\n"
        "**상태:** 업무 대기 중"
    )
    st.divider()
    st.markdown("📂 **데이터 및 컨텍스트 관리**")
    if uploaded_file is not None:
        st.success(f"파일 연동됨: `{uploaded_file.name}`")
    else:
        st.caption("참고용 텍스트 파일이 업로드되지 않았습니다.")
        
    st.divider()
    if st.button("🔄 대화 내용 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# 자동 실행형 추천 질문 칩 (Quick Prompt Chips) - key 추가 수정본
# =========================================================
st.markdown("💡 **자주 찾는 업무 및 추천 질문 (클릭 시 즉시 실행)**")
col1, col2, col3 = st.columns(3)

if "preset_query" not in st.session_state:
    st.session_state.preset_query = ""

with col1:
    if st.button("📊 최근 마케팅 회의록 요약", use_container_width=True, key="btn_summary"):
        st.session_state.preset_query = "최근 마케팅 회의록을 분석하고 핵심 업무 지시사항을 요약해줘."
        st.rerun()
with col2:
    if st.button("✉️ 마케팅 콘텐츠 기획 지원", use_container_width=True, key="btn_content"):
        st.session_state.preset_query = "데이터 기반의 새로운 마케팅 콘텐츠 기획 아이디어를 제안해줘."
        st.rerun()
with col3:
    if st.button("📈 주요 마케팅 지표 분석 점검", use_container_width=True, key="btn_metrics"):
        st.session_state.preset_query = "주요 마케팅 지표 분석 및 전략 수립을 도와줘."
        st.rerun()

st.markdown("")
# 폼 입력 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    
    # 칩이 선택되었다면 화면에 안내 표시
    if st.session_state.preset_query:
        st.info(f"선택된 추천 질문: '{st.session_state.preset_query}'")
        
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")

# [조건] 폼의 '실행' 버튼을 눌렀거나 OR '추천 질문 칩'이 클릭된 경우 모두 실행
if submit or st.session_state.preset_query:
    
    # 업로드 된 파일 처리
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    # 텍스트 소스 결정 (칩이 눌렸다면 칩의 내용, 아니면 녹음 확인)
    user_text = ""
    
    if st.session_state.preset_query:
        user_text = st.session_state.preset_query
        # 사용 후 초기화하여 중복 실행 방지
        st.session_state.preset_query = "" 
    elif audio_value is not None:
        # STT: 음성 → 텍스트
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", audio_value, "audio/wav")
        )
        user_text = transcript.text
    else:
        st.warning("음성을 녹음하시거나 추천 질문을 선택해주세요.")
        st.stop()

    # 시스템 메시지 구성
    system_content = sys_role
    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"
    input_messages = [{"role": "system", "content": system_content}]
    input_messages.append({"role": "user", "content": user_text})

    # LLM 호출 및 응답
    with st.spinner("답변을 준비하고 있습니다..."):
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages
        )
        answer = response.output_text

        # TTS 음성 재생 (앞서 다듬은 플레이어 상단 배치 구조 적용)
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=answer
        )
        audio_bytes = speech_response.read()
        
        st.caption("🔊 AI 비서 음성 답변")
        st.audio(io.BytesIO(audio_bytes), format="audio/mp3", autoplay=False)
        
        st.subheader("AI 답변")
        st.write(answer)
