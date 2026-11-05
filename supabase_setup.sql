-- =====================================================================
--  에이블(Able) Supabase 설정  ·  SQL Editor에 전체를 붙여 넣고 Run
--  여러 번 실행해도 안전합니다(이미 있으면 건너뜀/덮어씀).
-- =====================================================================

-- 1) 앱이 모든 데이터를 저장하는 표
create table if not exists public.able_state (
  k       text primary key,   -- 항목 이름 (tasks, users, messages::김 대리 ...)
  v       text not null,      -- 내용(JSON)
  updated text                -- 마지막 저장 시각
);

-- 2) 보안: 인터넷 공개 API(anon·authenticated)로는 읽거나 쓸 수 없게 잠금
--    앱은 DB 주인 계정(DATABASE_URL)으로 직접 연결하므로 그대로 동작합니다.
alter table public.able_state enable row level security;
revoke all on table public.able_state from anon, authenticated;

-- 3) Supabase 화면(Table Editor)에서 보기 쉬운 읽기 전용 보기(View)
--    팀 업무
create or replace view public.able_tasks with (security_invoker = true) as
select t->>'담당자'                      as "담당자",
       t->>'업무'                        as "업무",
       t->>'상태'                        as "상태",
       (t->'마감일'->>'__date__')::date   as "마감일",
       t->>'우선순위'                    as "우선순위",
       t->>'상세'                        as "상세"
from public.able_state s, jsonb_array_elements(s.v::jsonb) t
where s.k = 'tasks';

--    팀원 계정 (비밀번호는 보여 주지 않음)
create or replace view public.able_users with (security_invoker = true) as
select u->>'name' as "이름", u->>'role' as "직책", u->>'team' as "소속",
       u->>'email' as "이메일", u->>'created' as "가입일"
from public.able_state s, jsonb_array_elements(s.v::jsonb) u
where s.k = 'users';

--    일정
create or replace view public.able_schedule with (security_invoker = true) as
select (e->'일자'->>'__date__')::date as "일자", e->>'시간' as "시간", e->>'일정' as "일정",
       coalesce((e->>'소요')::int, 60) as "소요(분)"
from public.able_state s, jsonb_array_elements(s.v::jsonb) e
where s.k = 'schedule';

--    연락처
create or replace view public.able_contacts with (security_invoker = true) as
select c->>'이름' as "이름", c->>'이메일' as "이메일", c->>'직책' as "직책"
from public.able_state s, jsonb_array_elements(s.v::jsonb) c
where s.k = 'contacts';

--    보낸 메일
create or replace view public.able_sent_mail with (security_invoker = true) as
select m->>'발송 시각' as "발송 시각", m->>'받는 사람' as "받는 사람", m->>'제목' as "제목", m->>'방식' as "방식"
from public.able_state s, jsonb_array_elements(s.v::jsonb) m
where s.k = 'sent';

-- 보기(View)도 공개 API로는 볼 수 없게 잠금
revoke all on table public.able_tasks, public.able_users, public.able_schedule,
                    public.able_contacts, public.able_sent_mail from anon, authenticated;

-- 4) 확인: 아래 결과에 able_state와 able_ 보기 5개가 나오면 완료
select table_name, table_type
from information_schema.tables
where table_schema = 'public' and table_name like 'able%'
order by table_name;
