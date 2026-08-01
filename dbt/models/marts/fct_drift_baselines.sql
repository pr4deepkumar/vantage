-- Fact model computing baseline distribution statistics for Z-score operational drift detection

with spans as (
    select * from {{ ref('stg_spans') }}
    where execution_status = 'success'
)

select
    prompt_version,
    model_name,
    count(span_id) as sample_count,
    
    -- Latency distribution stats
    round(avg(latency_ms), 2) as mean_latency_ms,
    round(min(latency_ms), 2) as min_latency_ms,
    round(max(latency_ms), 2) as max_latency_ms,
    
    -- Cost distribution stats
    round(avg(token_cost_usd), 6) as mean_cost_usd,
    round(min(token_cost_usd), 6) as min_cost_usd,
    round(max(token_cost_usd), 6) as max_cost_usd,
    
    -- Token distribution stats
    round(avg(output_tokens), 1) as mean_output_tokens,
    round(min(output_tokens), 0) as min_output_tokens,
    round(max(output_tokens), 0) as max_output_tokens,
    
    -- Evaluator quality stats
    round(avg(judge_score), 2) as mean_judge_score
from spans
group by prompt_version, model_name
