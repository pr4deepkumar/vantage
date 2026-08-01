-- Fact model aggregating nested spans into trace-level execution summaries

with spans as (
    select * from {{ ref('stg_spans') }}
),

root_spans as (
    select
        span_id as trace_id,
        created_at,
        prompt_version,
        model_name,
        span_kind as root_span_kind,
        execution_status as root_status,
        error_type as root_error_type,
        user_feedback,
        judge_score,
        judge_coherence,
        judge_relevance,
        is_hallucination
    from spans
    where parent_span_id is null
),

span_aggregates as (
    select
        coalesce(parent_span_id, span_id) as trace_group_id,
        count(span_id) as total_span_count,
        sum(case when span_kind = 'llm' then 1 else 0 end) as llm_span_count,
        sum(case when span_kind = 'tool' then 1 else 0 end) as tool_span_count,
        sum(case when span_kind = 'agent' then 1 else 0 end) as agent_span_count,
        sum(input_tokens) as total_input_tokens,
        sum(output_tokens) as total_output_tokens,
        sum(total_tokens) as total_tokens,
        sum(token_cost_usd) as total_cost_usd,
        max(latency_ms) as trace_latency_ms
    from spans
    group by 1
)

select
    r.trace_id,
    r.created_at,
    r.prompt_version,
    r.model_name,
    r.root_span_kind,
    r.root_status,
    r.root_error_type,
    r.user_feedback,
    r.judge_score,
    r.judge_coherence,
    r.judge_relevance,
    r.is_hallucination,
    
    coalesce(a.total_span_count, 1) as total_span_count,
    coalesce(a.llm_span_count, 1) as llm_span_count,
    coalesce(a.tool_span_count, 0) as tool_span_count,
    coalesce(a.agent_span_count, 0) as agent_span_count,
    coalesce(a.total_input_tokens, 0) as total_input_tokens,
    coalesce(a.total_output_tokens, 0) as total_output_tokens,
    coalesce(a.total_tokens, 0) as total_tokens,
    coalesce(a.total_cost_usd, 0.0) as total_cost_usd,
    coalesce(a.trace_latency_ms, 0.0) as trace_latency_ms
from root_spans r
left join span_aggregates a on r.trace_id = a.trace_group_id
