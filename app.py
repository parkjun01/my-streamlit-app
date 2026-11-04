
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

# ── (추가) 기존 음성 폼의 결과 출력 + 대화 히스토리 저장 ──────────────
    st.session_state.messages.append({"role": "user", "content": transcript.text})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("user"):
        st.markdown(transcript.text)
    with st.chat_message("assistant"):
        st.markdown(answer)
        try:
            _speech = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=st.session_state.get("settings", {}).get("voice", "alloy"),
                input=answer[:4000])
            st.audio(_speech.content, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.caption(f"🔇 음성 재생 실패: {e}")


# =====================================================================
#  AI 개인비서 '에이블(Able)' 확장 기능
# =====================================================================
import hashlib
import uuid
from datetime import date, datetime, timedelta
import pandas as pd

ss = st.session_state
TODAY = date.today()

# 화면 표시 모델명 → 실제 호출 모델
MODEL_MAP = {"Luna 6": "gpt-4o-mini", "Luna 6 Pro": "gpt-4o"}
TTS_MODEL = "gpt-4o-mini-tts"
VOICES = ["alloy", "nova", "shimmer", "coral", "echo", "onyx", "fable"]
MENUS = ["🏠 홈", "💬 채팅", "📁 자료실", "✅ 내 작업", "👥 팀 업무 현황", "📧 이메일 관리", "⚙️ 설정"]
QUICK_PROMPTS = ["📝 회의록 요약해줘", "📅 오늘 일정 정리해줘",
                 "✅ 우선순위 높은 업무 알려줘", "📊 캠페인 성과 분석 방법 알려줘"]
CHAT_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "txt", "md", "csv", "docx"]
IMG_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
INTENT_LABEL = {"chat": "💬 일반 대화", "email": "📧 이메일 작성·발송", "calendar": "📅 캘린더 일정",
                "task": "✅ 업무 등록", "translate": "🌐 번역"}


# ---------------------------------------------------------------------
# 상태 초기화 (샘플 데이터 포함)
# ---------------------------------------------------------------------
def _defaults():
    d = TODAY
    return {
        "menu": MENUS[0],
        "settings": {"name": "김 대리", "team": "마케팅팀", "role": "Marketing Data Analyst",
                     "model": "Luna 6", "common_prompt": "", "voice": "alloy",
                     "tts_on": True, "autoplay": True, "stt_lang": "ko"},
        "tasks": [
            {"id": "t1", "업무": "SNS 광고 A/B 테스트 결과 분석", "담당자": "김 대리", "마감일": d, "상태": "진행 중", "우선순위": "높음"},
            {"id": "t2", "업무": "11월 캠페인 성과 리포트 작성", "담당자": "김 대리", "마감일": d + timedelta(days=1), "상태": "진행 중", "우선순위": "높음"},
            {"id": "t3", "업무": "신제품 런칭 보도자료 검토", "담당자": "김 대리", "마감일": d + timedelta(days=5), "상태": "대기", "우선순위": "보통"},
            {"id": "t4", "업무": "고객 설문 데이터 정제", "담당자": "김 대리", "마감일": d - timedelta(days=1), "상태": "완료", "우선순위": "보통"},
            {"id": "t5", "업무": "인플루언서 협업 리스트 정리", "담당자": "이 과장", "마감일": d + timedelta(days=2), "상태": "진행 중", "우선순위": "높음"},
            {"id": "t6", "업무": "연말 프로모션 예산안 작성", "담당자": "최 팀장", "마감일": d + timedelta(days=7), "상태": "대기", "우선순위": "높음"},
            {"id": "t7", "업무": "블로그 콘텐츠 3건 발행", "담당자": "박 사원", "마감일": d + timedelta(days=3), "상태": "진행 중", "우선순위": "보통"},
            {"id": "t8", "업무": "경쟁사 가격 모니터링", "담당자": "박 사원", "마감일": d - timedelta(days=2), "상태": "완료", "우선순위": "낮음"},
        ],
        "schedule": [
            {"일자": d, "시간": "09:30", "일정": "주간 마케팅 회의"},
            {"일자": d, "시간": "11:00", "일정": "광고대행사 미팅"},
            {"일자": d, "시간": "14:00", "일정": "캠페인 데이터 리뷰"},
            {"일자": d, "시간": "16:30", "일정": "팀장님 주간 보고"},
            {"일자": d + timedelta(days=1), "시간": "10:00", "일정": "신제품 기획 회의"},
        ],
        "inbox": [
            {"id": "m1", "from": "최 팀장", "addr": "choi@company.com", "date": f"{d} 08:40",
             "subject": "연말 프로모션 아이디어 요청",
             "body": "김 대리님, 금요일까지 연말 프로모션 아이디어 3가지를 정리해서 공유 부탁드립니다."},
            {"id": "m2", "from": "광고대행사 정 실장", "addr": "jung@agency.com", "date": f"{d} 07:55",
             "subject": "오늘 11시 미팅 자료 송부",
             "body": "안녕하세요, 오늘 미팅 때 논의할 4분기 매체 집행안을 미리 보내드립니다. 검토 부탁드립니다."},
            {"id": "m3", "from": "이 과장", "addr": "lee@company.com", "date": f"{d - timedelta(days=1)} 18:10",
             "subject": "인플루언서 후보 리스트 확인",
             "body": "후보 리스트 1차 정리했습니다. 데이터 기준으로 우선순위 의견 주시면 반영하겠습니다."},
        ],
        "sent": [],
        "library": [],
        "suggestions": [],
        "pending_prompt": None,
        "attach_ids": [],
        "chat_key": 0,
        "lib_key": 0,
        "last_audio": None,
        "briefing": "",
        "open_tools": False,
        "mail_to": "", "mail_subject": "", "mail_body": "",
        "notice": "",
    }

for _k, _v in _defaults().items():
    if _k not in ss:
        ss[_k] = _v


# ---------------------------------------------------------------------
# 공통 도우미 함수
# ---------------------------------------------------------------------
def model_id():
    return MODEL_MAP.get(ss.settings["model"], "gpt-4o-mini")


def to_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return TODAY


def dday(d):
    n = (d - TODAY).days
    return "D-day" if n == 0 else (f"D-{n}" if n > 0 else f"D+{-n}")


def base_system():
    s = ss.settings
    today_sch = "\n".join(f"- {e['시간']} {e['일정']}" for e in sorted(ss.schedule, key=lambda x: x["시간"])
                          if e["일자"] == TODAY) or "- 없음"
    my_open = "\n".join(f"- {t['업무']} (마감 {t['마감일']}, {t['상태']}, 우선순위 {t['우선순위']})"
                        for t in ss.tasks if t["담당자"] == s["name"] and t["상태"] != "완료") or "- 없음"
    prompt = (sys_role + f"\n사용자: {s['name']} ({s['team']}, {s['role']})\n오늘 날짜: {TODAY}"
              f"\n\n[오늘 일정]\n{today_sch}\n\n[사용자의 미완료 업무]\n{my_open}")
    if s["common_prompt"]:
        prompt += f"\n\n[공통 지침]\n{s['common_prompt']}"
    return prompt


def llm(messages, json_mode=False):
    kwargs = {"model": model_id(), "input": messages}
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
    return client.responses.create(**kwargs).output_text


def stream_llm(messages):
    # 실시간 스트리밍: 글자 조각(delta)을 받는 대로 내보냄
    stream = client.responses.create(model=model_id(), input=messages, stream=True)
    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta


def speak(text_, force=False):
    s = ss.settings
    if not text_ or (not s["tts_on"] and not force):
        return
    try:
        sp = client.audio.speech.create(model=TTS_MODEL, voice=s["voice"], input=text_[:4000])
        st.audio(sp.content, format="audio/mp3", autoplay=s["autoplay"] or force)
    except Exception as e:
        st.caption(f"🔇 음성 변환 실패: {e}")


def stt(audio_bytes):
    r = client.audio.transcriptions.create(
        model="whisper-1", file=("audio.wav", audio_bytes, "audio/wav"),
        language=ss.settings["stt_lang"])
    return r.text


# ---------------------------------------------------------------------
# 자료실 / RAG
# ---------------------------------------------------------------------
def extract_text(name, data):
    ext = name.rsplit(".", 1)[-1].lower()
    try:
        if ext in ("txt", "md", "csv", "json"):
            return data.decode("utf-8", errors="ignore")
        if ext == "pdf":
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
        if ext == "docx":
            import docx
            return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    except Exception:
        pass
    return ""


def add_to_library(name, data, source):
    for f in ss.library:
        if f["name"] == name and f["size"] == len(data):
            return f
    text_ = extract_text(name, data)
    rec = {"id": uuid.uuid4().hex[:8], "name": name, "data": data, "text": text_,
           "source": source, "size": len(data), "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "rag": bool(text_)}
    ss.library.append(rec)
    return rec


def rag_context(query):
    # 간단한 키워드 기반 검색: 500자 단위로 나눠 질문 단어가 많이 나온 조각 3개 선택
    words = [w for w in query.split() if len(w) >= 2]
    chunks = []
    for f in ss.library:
        if not (f["rag"] and f["text"]):
            continue
        for i in range(0, len(f["text"]), 500):
            c = f["text"][i:i + 500]
            score = sum(c.count(w) for w in words)
            if score:
                chunks.append((score, f["name"], c))
    chunks.sort(reverse=True)
    top = chunks[:3]
    return "\n\n".join(f"[출처: {n}]\n{c}" for _, n, c in top), [n for _, n, _ in top]


def file_parts(f):
    # 멀티모달 입력: 이미지·PDF는 원본 그대로, 문서는 추출한 텍스트로 전달
    ext = f["name"].rsplit(".", 1)[-1].lower()
    b64 = base64.b64encode(f["data"]).decode()
    if ext in IMG_MIME:
        return [{"type": "input_image", "image_url": f"data:{IMG_MIME[ext]};base64,{b64}"}]
    if ext == "pdf":
        return [{"type": "input_file", "filename": f["name"],
                 "file_data": f"data:application/pdf;base64,{b64}"}]
    if f["text"]:
        return [{"type": "input_text", "text": f"[첨부: {f['name']}]\n{f['text'][:15000]}"}]
    return []


# ---------------------------------------------------------------------
# 의도 파악(라우팅) 및 외부 도구
# ---------------------------------------------------------------------
ROUTER_PROMPT = """너는 업무 비서의 라우터다. 사용자 요청을 분석해 JSON으로만 답하라.
intent는 chat | email | calendar | task | translate 중 하나.
- email: 이메일 작성/발송 요청. args: {"to": 받는 사람, "subject": 제목, "body": 정중한 업무 메일 본문}
- calendar: 일정 추가/조율. args: {"date": "YYYY-MM-DD", "time": "HH:MM", "title": 일정명}
- task: 업무 등록. args: {"title": 업무명, "owner": 담당자, "due": "YYYY-MM-DD", "priority": "높음|보통|낮음"}
- translate: 번역 요청. args: {"target_lang": 언어}
- chat: 그 외 대화·요약·분석. args: {}
오늘 날짜: {today}
형식: {"intent": "...", "args": {...}}"""


def route(prompt):
    try:
        out = llm([{"role": "system", "content": ROUTER_PROMPT.replace("{today}", str(TODAY))},
                   {"role": "user", "content": prompt}], json_mode=True)
        r = json.loads(out)
        if r.get("intent") in INTENT_LABEL:
            r.setdefault("args", {})
            return r
    except Exception:
        pass
    # LLM 라우팅 실패 시 키워드 규칙으로 대체
    rules = {"email": ["메일"], "calendar": ["일정 추가", "일정 잡", "캘린더"],
             "task": ["업무 등록", "할 일 추가"], "translate": ["번역"]}
    for intent, words in rules.items():
        if any(w in prompt for w in words):
            return {"intent": intent, "args": {}}
    return {"intent": "chat", "args": {}}


def tool_email(a):
    mail = {"받는 사람": a.get("to") or "(미지정)", "제목": a.get("subject") or "(제목 없음)",
            "본문": a.get("body") or "", "발송 시각": datetime.now().strftime("%Y-%m-%d %H:%M")}
    ss.sent.append(mail)
    return (f"이메일 발송 완료(모의) → 받는 사람: {mail['받는 사람']}, 제목: {mail['제목']}\n본문:\n{mail['본문']}",
            [mail])


def tool_calendar(a):
    d = to_date(a.get("date"))
    t = a.get("time") or "10:00"
    try:
        datetime.strptime(t, "%H:%M")
    except Exception:
        t = "10:00"
    busy = {e["시간"] for e in ss.schedule if e["일자"] == d}
    note = ""
    while t in busy:  # 겹치면 1시간씩 미뤄 자동 조율
        h, m = map(int, t.split(":"))
        t, note = f"{(h + 1) % 24:02d}:{m:02d}", " (기존 일정과 겹쳐 시간을 자동 조정함)"
    ss.schedule.append({"일자": d, "시간": t, "일정": a.get("title") or "새 일정"})
    day = sorted([e for e in ss.schedule if e["일자"] == d], key=lambda x: x["시간"])
    return f"{d} {t} '{a.get('title') or '새 일정'}' 일정 등록 완료{note}", day


def tool_task(a):
    t = {"id": uuid.uuid4().hex[:8], "업무": a.get("title") or "새 업무",
         "담당자": a.get("owner") or ss.settings["name"], "마감일": to_date(a.get("due")),
         "상태": "대기", "우선순위": a.get("priority") if a.get("priority") in ("높음", "보통", "낮음") else "보통"}
    ss.tasks.append(t)
    rows = [{k: v for k, v in x.items() if k != "id"} for x in ss.tasks if x["담당자"] == t["담당자"]]
    return f"업무 등록 완료: {t['업무']} (담당 {t['담당자']}, 마감 {t['마감일']})", rows


TOOLS = {"email": ("📧 이메일 도구", tool_email),
         "calendar": ("📅 캘린더 도구", tool_calendar),
         "task": ("✅ 업무 관리 도구", tool_task)}


def make_suggestions(q, a):
    try:
        out = llm([{"role": "system", "content": '대화 흐름에 이어서 사용자가 물어볼 만한 짧은 후속 질문 3개를 '
                                                 'JSON으로 만들어라. 형식: {"questions": ["...", "...", "..."]}'},
                   {"role": "user", "content": f"질문: {q}\n답변: {a[:2000]}"}], json_mode=True)
        return json.loads(out).get("questions", [])[:3]
    except Exception:
        return []


# ---------------------------------------------------------------------
# 콜백 함수 (버튼 클릭 시 실행)
# ---------------------------------------------------------------------
def go(page, prompt=None, open_tools=False):
    ss.menu = page
    ss.open_tools = open_tools
    if prompt:
        ss.pending_prompt = prompt


def set_prompt(q):
    ss.pending_prompt = q


def new_chat():
    ss.messages = []
    ss.suggestions = []
    ss.attach_ids = []
    ss.chat_key += 1
    ss.menu = "💬 채팅"


def toggle_task(tid):
    for t in ss.tasks:
        if t["id"] == tid:
            t["상태"] = "완료" if ss[f"chk_{tid}"] else "진행 중"


# ---------------------------------------------------------------------
# 채팅: 출력 렌더링 / 한 턴 처리
# ---------------------------------------------------------------------
def render_extras(i, m):
    # Rich Output: 표, 다운로드, 자료실 저장, 피드백, 복사
    if m.get("table"):
        st.dataframe(pd.DataFrame(m["table"]), hide_index=True)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.download_button("⬇️ 다운로드", m["content"], file_name=f"able_answer_{i}.md", key=f"dl_{i}")
    with c2:
        if st.button("💾 자료실 저장", key=f"save_{i}"):
            add_to_library(f"AI답변_{datetime.now():%m%d_%H%M%S}.md", m["content"].encode("utf-8"), "생성")
            st.toast("자료실에 저장했습니다.")
    with c3:
        st.feedback("thumbs", key=f"fb_{i}")
    with st.expander("📋 복사하기"):
        st.code(m["content"], language="markdown")


def handle_turn(prompt, files):
    with st.chat_message("user"):
        st.markdown(prompt)
        if files:
            st.caption("📎 " + ", ".join(f["name"] for f in files))

    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages[-20:]]
    ss.messages.append({"role": "user", "content": prompt, "files": [f["name"] for f in files]})

    with st.chat_message("assistant"):
        try:
            # 에이전트 실행 상태 시각화
            with st.status("에이블이 생각하는 중...", expanded=True) as status:
                st.write("🔍 1단계: 요청 의도 파악")
                recent = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history[-4:])
                r = route(f"[최근 대화]\n{recent}\n\n[현재 요청]\n{prompt}")
                intent, args = r["intent"], r["args"]
                st.write(f"➡️ 판독 결과: **{INTENT_LABEL[intent]}**")

                sys_c, table = base_system(), None
                if intent in TOOLS:
                    name, fn = TOOLS[intent]
                    st.write(f"🛠️ 2단계: {name} 실행")
                    result, table = fn(args)
                    st.write(f"✅ {result.splitlines()[0]}")
                    sys_c += f"\n\n[도구 실행 결과]\n{result}\n위 결과를 사용자에게 간결하게 보고하라."
                elif intent == "translate":
                    sys_c += f"\n\n사용자가 번역을 요청했다. 대상 언어: {args.get('target_lang', '영어')}. 번역문을 먼저 제시하라."

                st.write("📚 3단계: 자료실(RAG) 검색")
                ctx, srcs = rag_context(prompt)
                if ctx:
                    sys_c += f"\n\n[자료실 참고 자료]\n{ctx}\n참고한 경우 출처 파일명을 밝혀라."
                    st.write("참고 문서: " + ", ".join(sorted(set(srcs))))
                else:
                    st.write("관련 자료 없음")
                if files:
                    st.write(f"📎 첨부 파일 {len(files)}개 분석")
                status.update(label="분석 완료! 답변을 작성합니다.", state="complete", expanded=False)

            content = [{"type": "input_text", "text": prompt}]
            for f in files:
                content += file_parts(f)
            msgs = [{"role": "system", "content": sys_c}] + history + [{"role": "user", "content": content}]
            answer_ = st.write_stream(stream_llm(msgs))
        except Exception as e:
            # Error Fallback: 앱이 멈추지 않도록 안내 후 재시도 버튼 제공
            ss.messages.pop()
            st.error(f"⚠️ 답변 생성 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.\n\n`{e}`")
            st.button("🔄 다시 시도", on_click=set_prompt, args=(prompt,), key="retry")
            return

        msg = {"role": "assistant", "content": answer_ or "", "table": table}
        ss.messages.append(msg)
        render_extras(len(ss.messages) - 1, msg)
        speak(answer_)

    ss.suggestions = make_suggestions(prompt, answer_ or "")
    ss.attach_ids = []
    ss.chat_key += 1  # 다음 실행 때 업로더·녹음기 초기화


# ---------------------------------------------------------------------
# 페이지: 홈(대시보드)
# ---------------------------------------------------------------------
def page_home():
    s = ss.settings
    st.header(f"🏠 {s['name']}님, 좋은 하루입니다!")
    st.caption(TODAY.strftime("%Y년 %m월 %d일"))

    total = len(ss.tasks)
    done = sum(t["상태"] == "완료" for t in ss.tasks)
    doing = sum(t["상태"] == "진행 중" for t in ss.tasks)
    urgent = [t for t in ss.tasks if t["상태"] != "완료" and t["마감일"] <= TODAY + timedelta(days=2)]

    c = st.columns(4)
    c[0].metric("📋 전체 업무", total)
    c[1].metric("✅ 완료 업무", done)
    c[2].metric("🔄 진행 중", doing)
    c[3].metric("⏰ 마감 임박", len(urgent))
    st.progress(done / total if total else 0, text=f"완료율 {done / total * 100:.0f}%" if total else "업무 없음")

    left, right = st.columns(2)
    with left:
        st.subheader("📅 오늘의 일정")
        now = datetime.now().strftime("%H:%M")
        today = sorted([e for e in ss.schedule if e["일자"] == TODAY], key=lambda x: x["시간"])
        for e in today:
            mark = "✔️" if e["시간"] < now else "🕒"
            st.markdown(f"{mark} **{e['시간']}** {e['일정']}")
        if not today:
            st.caption("오늘 일정이 없습니다.")
    with right:
        st.subheader("⏰ 마감 임박 업무")
        order = {"높음": 0, "보통": 1, "낮음": 2}
        for t in sorted(urgent, key=lambda x: (x["마감일"], order[x["우선순위"]])):
            st.markdown(f"**[{dday(t['마감일'])}]** {t['업무']} · {t['담당자']} · 우선순위 {t['우선순위']}")
        if not urgent:
            st.caption("마감 임박 업무가 없습니다.")

    st.subheader("📊 상태별 업무")
    st.bar_chart(pd.DataFrame(ss.tasks)["상태"].value_counts())

    st.subheader("🤖 AI 업무 브리핑")
    if st.button("오늘의 브리핑 받기"):
        with st.spinner("오늘의 우선순위를 분석하고 있습니다..."):
            try:
                ss.briefing = llm([{"role": "system", "content": base_system()},
                                   {"role": "user", "content": "오늘 일정과 내 미완료 업무를 보고 마감일·우선순위를 고려해 "
                                    "가장 효율적인 업무 순서를 번호 목록으로 제안하고, 업무마다 한 줄 가이드를 붙여줘."}])
            except Exception as e:
                ss.briefing = f"⚠️ 브리핑 생성 중 오류가 발생했습니다. 다시 시도해 주세요. ({e})"
    if ss.briefing:
        st.info(ss.briefing)

    st.subheader("⚡ 빠른 실행")
    q = st.columns(4)
    q[0].button("🎤 음성 비서", on_click=go, args=("💬 채팅", None, True))
    q[1].button("📝 회의록 요약", on_click=go, args=("💬 채팅", None, True))
    q[2].button("📧 이메일 작성", on_click=go, args=("📧 이메일 관리",))
    q[3].button("📅 일정 정리", on_click=go, args=("💬 채팅", "오늘 일정과 마감 임박 업무를 시간순으로 정리해줘"))


# ---------------------------------------------------------------------
# 페이지: 채팅
# ---------------------------------------------------------------------
def page_chat():
    st.header("💬 채팅")

    # 멀티턴 히스토리 출력
    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_extras(i, m)
            elif m.get("files"):
                st.caption("📎 " + ", ".join(m["files"]))

    chat_area = st.container()  # 새 대화는 히스토리 바로 아래에 표시

    # 입력 도구
    with st.expander("📎 파일 첨부 · 🎤 음성 입력", expanded=ss.open_tools):
        ups = st.file_uploader("PDF, 이미지, 문서 파일을 끌어다 놓으세요", type=CHAT_TYPES,
                               accept_multiple_files=True, key=f"chat_up_{ss.chat_key}")
        audio = st.audio_input("마이크로 말씀하세요", key=f"chat_audio_{ss.chat_key}")
    attached = [f for f in ss.library if f["id"] in ss.attach_ids]
    if attached:
        st.caption("📁 자료실에서 첨부됨: " + ", ".join(f["name"] for f in attached))

    st.caption("⚡ 자주 쓰는 질문")
    cols = st.columns(len(QUICK_PROMPTS))
    for col, qp in zip(cols, QUICK_PROMPTS):
        col.button(qp, key=f"qp_{qp}", on_click=set_prompt, args=(qp,))

    prompt = st.chat_input("에이블에게 무엇이든 요청하세요")

    if ss.pending_prompt:
        prompt, ss.pending_prompt = ss.pending_prompt, None
    if audio is not None:
        audio_bytes = audio.getvalue()
        h = hashlib.md5(audio_bytes).hexdigest()
        if h != ss.last_audio:
            ss.last_audio = h
            try:
                prompt = stt(audio_bytes)
            except Exception as e:
                st.error(f"음성 인식에 실패했습니다. 다시 녹음해 주세요. ({e})")

    if prompt:
        files = [add_to_library(u.name, u.getvalue(), "업로드") for u in (ups or [])]
        files += [f for f in attached if f not in files]
        ss.open_tools = False
        with chat_area:
            handle_turn(prompt, files)

    if ss.suggestions:
        st.caption("💡 이어서 물어보기")
        for j, sq in enumerate(ss.suggestions):
            st.button(sq, key=f"sg_{j}", on_click=set_prompt, args=(sq,))


# ---------------------------------------------------------------------
# 페이지: 자료실
# ---------------------------------------------------------------------
def lib_upload():
    for u in ss.get(f"lib_up_{ss.lib_key}") or []:
        add_to_library(u.name, u.getvalue(), "업로드")
    ss.lib_key += 1


def lib_delete(fid):
    ss.library = [f for f in ss.library if f["id"] != fid]
    ss.attach_ids = [x for x in ss.attach_ids if x != fid]


def lib_rag(fid):
    for f in ss.library:
        if f["id"] == fid:
            f["rag"] = ss[f"rag_{fid}"]


def lib_attach(fid):
    if fid not in ss.attach_ids:
        ss.attach_ids.append(fid)
    ss.menu = "💬 채팅"


LIB_DIR = "library"  # Colab에서는 /content/library


def lib_import_folder():
    os.makedirs(LIB_DIR, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(LIB_DIR)):
        if fn.rsplit(".", 1)[-1].lower() in CHAT_TYPES:
            with open(os.path.join(LIB_DIR, fn), "rb") as fp:
                add_to_library(fn, fp.read(), "폴더")
            n += 1
    ss.notice = f"📂 폴더에서 {n}개 파일을 불러왔습니다." if n else f"📂 '{LIB_DIR}' 폴더에 파일이 없습니다."


def lib_add_text():
    body = ss.get("lib_text_body", "")
    if not body.strip():
        ss.notice = "⚠️ 내용을 입력해 주세요."
        return
    name = ss.get("lib_text_name", "").strip() or f"메모_{datetime.now():%m%d_%H%M%S}"
    if not name.endswith(".txt"):
        name += ".txt"
    add_to_library(name, body.encode("utf-8"), "직접 입력")
    ss.lib_text_name = ""
    ss.lib_text_body = ""
    ss.notice = f"✍️ '{name}'을(를) 자료실에 추가했습니다."


def page_library():
    st.header("📁 자료실")
    if ss.notice:
        st.success(ss.notice)
        ss.notice = ""

    tab1, tab2, tab3 = st.tabs(["⬆️ 파일 업로드", "📂 폴더에서 가져오기", "✍️ 텍스트 붙여넣기"])
    with tab1:
        st.file_uploader("자료 추가", type=CHAT_TYPES, accept_multiple_files=True, key=f"lib_up_{ss.lib_key}")
        st.button("➕ 자료실에 추가", on_click=lib_upload)
        st.caption("업로드 오류가 나면 '폴더에서 가져오기'나 '텍스트 붙여넣기'를 사용하세요.")
    with tab2:
        os.makedirs(LIB_DIR, exist_ok=True)
        st.write(f"이 폴더에 파일을 넣은 뒤 버튼을 누르세요: `{os.path.abspath(LIB_DIR)}`")
        found = [fn for fn in sorted(os.listdir(LIB_DIR)) if fn.rsplit(".", 1)[-1].lower() in CHAT_TYPES]
        st.caption("폴더 안 파일: " + (", ".join(found) if found else "없음"))
        st.button("📂 폴더 파일 불러오기", on_click=lib_import_folder)
    with tab3:
        st.text_input("파일 이름", key="lib_text_name", placeholder="예: 11월_마케팅회의록")
        st.text_area("내용", key="lib_text_body", height=200, placeholder="회의록이나 참고 자료를 붙여 넣으세요.")
        st.button("✍️ 자료실에 저장", on_click=lib_add_text)

    query = st.text_input("🔍 파일명·내용 검색")
    files = [f for f in ss.library if not query or query in f["name"] or query in f["text"]]
    st.caption(f"총 {len(files)}개 · RAG 사용 {sum(f['rag'] for f in files)}개")

    for f in files:
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([4, 1.3, 1.3, 1.5, 0.8])
            c1.markdown(f"**{f['name']}**  \n{f['source']} · {f['created']} · {f['size'] / 1024:.1f}KB")
            c2.checkbox("RAG", value=f["rag"], key=f"rag_{f['id']}", on_change=lib_rag, args=(f["id"],),
                        disabled=not f["text"], help="텍스트를 읽을 수 있는 파일만 검색에 사용됩니다.")
            c3.download_button("⬇️", f["data"], file_name=f["name"], key=f"ldl_{f['id']}")
            c4.button("💬 대화에 활용", key=f"use_{f['id']}", on_click=lib_attach, args=(f["id"],))
            c5.button("🗑️", key=f"del_{f['id']}", on_click=lib_delete, args=(f["id"],))
    if not files:
        st.info("자료가 없습니다. 위의 세 가지 방법 중 하나로 자료를 추가해 보세요.")


# ---------------------------------------------------------------------
# 페이지: 내 작업 / 팀 업무 현황
# ---------------------------------------------------------------------
def page_my_tasks():
    me = ss.settings["name"]
    st.header("✅ 내 작업")
    view = st.radio("보기", ["전체", "미완료", "완료"], horizontal=True)
    mine = sorted([t for t in ss.tasks if t["담당자"] == me], key=lambda x: x["마감일"])
    if view == "미완료":
        mine = [t for t in mine if t["상태"] != "완료"]
    elif view == "완료":
        mine = [t for t in mine if t["상태"] == "완료"]

    for t in mine:
        c1, c2, c3 = st.columns([5, 1.5, 1.5])
        c1.checkbox(t["업무"], value=t["상태"] == "완료", key=f"chk_{t['id']}",
                    on_change=toggle_task, args=(t["id"],))
        c2.markdown(f"`{dday(t['마감일'])}` {t['마감일']}")
        c3.markdown(f"{t['상태']} · {t['우선순위']}")
    if not mine:
        st.caption("해당하는 업무가 없습니다.")

    with st.form("add_task", clear_on_submit=True):
        st.subheader("➕ 업무 추가")
        c1, c2, c3 = st.columns([3, 1.5, 1.5])
        title = c1.text_input("업무명")
        due = c2.date_input("마감일", TODAY)
        pr = c3.selectbox("우선순위", ["높음", "보통", "낮음"], index=1)
        if st.form_submit_button("추가") and title:
            ss.tasks.append({"id": uuid.uuid4().hex[:8], "업무": title, "담당자": me,
                             "마감일": due, "상태": "대기", "우선순위": pr})
            st.rerun()


def page_team():
    st.header("👥 팀 업무 현황")
    df = pd.DataFrame(ss.tasks).drop(columns="id")
    owners = sorted(df["담당자"].unique())
    c1, c2 = st.columns(2)
    sel = c1.multiselect("담당자", owners, default=owners)
    status_ = c2.selectbox("상태", ["전체", "대기", "진행 중", "완료"])
    view = df[df["담당자"].isin(sel)]
    if status_ != "전체":
        view = view[view["상태"] == status_]
    st.dataframe(view.sort_values("마감일"), hide_index=True)
    st.subheader("📊 담당자별 업무 상태")
    if not view.empty:
        st.bar_chart(pd.crosstab(view["담당자"], view["상태"]))


# ---------------------------------------------------------------------
# 페이지: 이메일 관리
# ---------------------------------------------------------------------
def draft_reply(mid):
    m = next(x for x in ss.inbox if x["id"] == mid)
    try:
        body = llm([{"role": "system", "content": base_system() + "\n받은 이메일에 대한 정중한 답장 본문만 작성하라."},
                    {"role": "user", "content": f"보낸 사람: {m['from']}\n제목: {m['subject']}\n\n{m['body']}"}])
    except Exception as e:
        body = f"(초안 생성 실패: {e})"
    ss.mail_to, ss.mail_subject, ss.mail_body = m["addr"], "RE: " + m["subject"], body
    ss.notice = "✍️ 답장 초안이 '메일 작성' 탭에 준비되었습니다."


def mail_to_task(mid):
    m = next(x for x in ss.inbox if x["id"] == mid)
    ss.tasks.append({"id": uuid.uuid4().hex[:8], "업무": f"[메일] {m['subject']}", "담당자": ss.settings["name"],
                     "마감일": TODAY + timedelta(days=2), "상태": "대기", "우선순위": "보통"})
    ss.notice = "📌 '내 작업'에 업무로 등록했습니다."


def ai_compose():
    try:
        ss.mail_body = llm([{"role": "system", "content": base_system() + "\n주어진 제목과 메모로 정중한 업무 메일 본문을 작성하라."},
                            {"role": "user", "content": f"받는 사람: {ss.mail_to}\n제목: {ss.mail_subject}\n메모: {ss.mail_body}"}])
    except Exception as e:
        ss.notice = f"⚠️ AI 작성 실패: {e}"


def send_mail():
    if not ss.mail_to or not ss.mail_subject:
        ss.notice = "⚠️ 받는 사람과 제목을 입력해 주세요."
        return
    ss.sent.append({"받는 사람": ss.mail_to, "제목": ss.mail_subject, "본문": ss.mail_body,
                    "발송 시각": datetime.now().strftime("%Y-%m-%d %H:%M")})
    ss.mail_to = ss.mail_subject = ss.mail_body = ""
    ss.notice = "📤 메일을 발송했습니다. (모의 발송)"


def page_email():
    st.header("📧 이메일 관리")
    if ss.notice:
        st.success(ss.notice)
        ss.notice = ""
    tab1, tab2, tab3 = st.tabs([f"📥 받은편지함 ({len(ss.inbox)})", "✍️ 메일 작성", f"📤 보낸편지함 ({len(ss.sent)})"])
    with tab1:
        for m in ss.inbox:
            with st.expander(f"**{m['subject']}** — {m['from']} · {m['date']}"):
                st.write(m["body"])
                c1, c2 = st.columns(2)
                c1.button("🤖 AI 답장 초안", key=f"rep_{m['id']}", on_click=draft_reply, args=(m["id"],))
                c2.button("📌 업무로 등록", key=f"tsk_{m['id']}", on_click=mail_to_task, args=(m["id"],))
    with tab2:
        st.text_input("받는 사람", key="mail_to")
        st.text_input("제목", key="mail_subject")
        st.text_area("본문 (메모만 적고 AI 작성 도움을 눌러도 됩니다)", key="mail_body", height=250)
        c1, c2 = st.columns(2)
        c1.button("🤖 AI 작성 도움", on_click=ai_compose)
        c2.button("📤 보내기", on_click=send_mail, type="primary")
    with tab3:
        if ss.sent:
            st.dataframe(pd.DataFrame(ss.sent), hide_index=True)
        else:
            st.caption("보낸 메일이 없습니다.")


# ---------------------------------------------------------------------
# 페이지: 설정
# ---------------------------------------------------------------------
def page_settings():
    s = ss.settings
    st.header("⚙️ 설정")
    if ss.notice:
        st.success(ss.notice)
        ss.notice = ""
    with st.form("settings_form"):
        st.subheader("👤 사용자 정보")
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("이름", s["name"])
        team = c2.text_input("소속", s["team"])
        role = c3.text_input("직무", s["role"])

        st.subheader("🧠 AI 모델")
        models = list(MODEL_MAP)
        model = st.selectbox("모델", models, index=models.index(s["model"]),
                             help=" / ".join(f"{k} → {v}" for k, v in MODEL_MAP.items()))
        common = st.text_area("공통 프롬프트 (모든 답변에 적용)", s["common_prompt"],
                              placeholder="예: 답변은 항상 3줄 요약으로 시작해 주세요.")

        st.subheader("🔊 음성 설정")
        c4, c5, c6, c7 = st.columns(4)
        voice = c4.selectbox("TTS 목소리", VOICES, index=VOICES.index(s["voice"]))
        tts_on = c5.toggle("답변 음성 재생", s["tts_on"])
        autoplay = c6.toggle("자동 재생", s["autoplay"])
        langs = {"한국어": "ko", "English": "en", "日本語": "ja"}
        lang = c7.selectbox("STT 언어", list(langs), index=list(langs.values()).index(s["stt_lang"]))

        if st.form_submit_button("💾 저장"):
            s.update({"name": name, "team": team, "role": role, "model": model, "common_prompt": common,
                      "voice": voice, "tts_on": tts_on, "autoplay": autoplay, "stt_lang": langs[lang]})
            ss.notice = "설정이 저장되었습니다."
            st.rerun()

    if st.button("🔊 목소리 미리 듣기"):
        speak(f"안녕하세요, {s['name']}님. 에이블입니다. 오늘도 함께 하겠습니다.", force=True)


# ---------------------------------------------------------------------
# 사이드바 + 페이지 이동
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 에이블 (Able)")
    st.caption(f"{ss.settings['name']} · {ss.settings['team']}")
    st.button("➕ 새 대화", on_click=new_chat, type="primary")
    st.radio("메뉴", MENUS, key="menu", label_visibility="collapsed")
    st.divider()
    st.caption(f"🧠 모델: {ss.settings['model']}")
    st.caption(f"💬 대화 {len(ss.messages)}개 · 📁 자료 {len(ss.library)}개")

PAGES = {"🏠 홈": page_home, "💬 채팅": page_chat, "📁 자료실": page_library,
         "✅ 내 작업": page_my_tasks, "👥 팀 업무 현황": page_team,
         "📧 이메일 관리": page_email, "⚙️ 설정": page_settings}

st.divider()
PAGES[ss.menu]()
