-- Daily rollup model for LLM token spend, request volumes, and model cost allocation

with spans as (
    select * from {{ ref('stg_spans') }}
)

select
    cast(created_at as date) as date_day,
    model_name,
    prompt_version,
    count(span_id) as total_span_calls,
    sum(case when execution_status = 'error' then 1 else 0 end) as total_errors,
    sum(input_tokens) as daily_input_tokens,
    sum(output_tokens) as daily_output_tokens,
    sum(total_tokens) as daily_total_tokens,
    round(sum(token_cost_usd), 4) as daily_cost_usd,
    round(avg(latency_ms), 2) as daily_avg_latency_ms
from spans
group by 1, 2, 3
