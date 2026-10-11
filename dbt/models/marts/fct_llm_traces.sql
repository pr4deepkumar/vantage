-- Fact model aggregating nested spans into trace-level execution summaries.
-- Model and quality come from the trace's LLM spans; agent, chain and tool spans are pipeline steps.

with spans as (
    select * from {{ ref('stg_spans') }}
),

root_spans as (
    select
        span_id as trace_id,
        created_at,
        prompt_version,
        span_kind as root_span_kind,
        execution_status as root_status,
        error_type as root_error_type
    from spans
    where parent_span_id is null
),

span_aggregates as (
    select
        coalesce(parent_span_id, span_id) as trace_group_id,
        count(span_id) as total_span_count,
        sum(case when span_kind = 'llm' then 1 else 0 end) as llm_span_count,
        sum(case when span_kind = 'tool' then 1 else 0 end) as tool_span_count,
        sum(case when span_kind in ('agent', 'chain') then 1 else 0 end) as step_span_count,
        max(case when span_kind = 'llm' then model_name end) as model_name,
        sum(input_tokens) as total_input_tokens,
        sum(output_tokens) as total_output_tokens,
        sum(total_tokens) as total_tokens,
        sum(token_cost_usd) as total_cost_usd,
        max(latency_ms) as trace_latency_ms,
        avg(case when span_kind = 'llm' then ttft_ms end) as avg_ttft_ms,
        max(user_feedback) as user_feedback,
        avg(case when span_kind = 'llm' then judge_score end) as judge_score,
        avg(case when span_kind = 'llm' then judge_coherence end) as judge_coherence,
        avg(case when span_kind = 'llm' then judge_relevance end) as judge_relevance,
        max(case when span_kind = 'llm' then is_hallucination end) as is_hallucination
    from spans
    group by 1
)

select
    r.trace_id,
    r.created_at,
    r.prompt_version,
    coalesce(a.model_name, 'no-llm-call') as model_name,
    r.root_span_kind,
    r.root_status,
    r.root_error_type,
    a.user_feedback,
    a.judge_score,
    a.judge_coherence,
    a.judge_relevance,
    a.is_hallucination,

    coalesce(a.total_span_count, 1) as total_span_count,
    coalesce(a.llm_span_count, 0) as llm_span_count,
    coalesce(a.tool_span_count, 0) as tool_span_count,
    coalesce(a.step_span_count, 0) as step_span_count,
    coalesce(a.total_input_tokens, 0) as total_input_tokens,
    coalesce(a.total_output_tokens, 0) as total_output_tokens,
    coalesce(a.total_tokens, 0) as total_tokens,
    coalesce(a.total_cost_usd, 0.0) as total_cost_usd,
    coalesce(a.trace_latency_ms, 0.0) as trace_latency_ms,
    a.avg_ttft_ms
from root_spans r
left join span_aggregates a on r.trace_id = a.trace_group_id
