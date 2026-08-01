-- Dimensional model for side-by-side Prompt Version A/B performance comparison

with traces as (
    select * from {{ ref('fct_llm_traces') }}
)

select
    prompt_version,
    model_name,
    count(trace_id) as total_requests,
    
    -- Error & status metrics
    sum(case when root_status = 'error' then 1 else 0 end) as error_count,
    round(100.0 * sum(case when root_status = 'error' then 1 else 0 end) / count(trace_id), 2) as error_rate_pct,
    
    -- Cost & latency metrics
    round(avg(total_cost_usd), 6) as avg_cost_per_request_usd,
    round(sum(total_cost_usd), 4) as total_spend_usd,
    round(avg(trace_latency_ms), 2) as avg_latency_ms,
    round(avg(total_tokens), 1) as avg_total_tokens,
    
    -- Quality judge metrics
    round(avg(judge_coherence), 2) as avg_coherence_score,
    round(avg(judge_relevance), 2) as avg_relevance_score,
    round(100.0 * sum(is_hallucination) / nullif(count(judge_score), 0), 2) as hallucination_rate_pct,
    
    -- User feedback metrics
    sum(case when user_feedback = 'up' then 1 else 0 end) as positive_feedback_count,
    sum(case when user_feedback = 'down' then 1 else 0 end) as negative_feedback_count,
    round(100.0 * sum(case when user_feedback = 'up' then 1 else 0 end) / nullif(count(user_feedback), 0), 2) as positive_feedback_pct
from traces
group by prompt_version, model_name
