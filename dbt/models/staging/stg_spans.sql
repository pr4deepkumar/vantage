-- Staging model for raw LLM telemetry spans
-- Cleans timestamps, computes total tokens, and standardizes nulls

with source as (
    select * from {{ source('main', 'traces') }}
),

renamed as (
    select
        cast(id as string) as span_id,
        cast(timestamp as timestamp) as created_at,
        coalesce(prompt_version, 'v1.0.0') as prompt_version,
        coalesce(model, 'unknown-model') as model_name,
        cast(parent_id as string) as parent_span_id,
        coalesce(span_kind, 'llm') as span_kind,
        
        -- Payload fields
        input as span_input,
        output as span_output,
        rag_context,
        
        -- Quantitative metrics
        coalesce(input_tokens, 0) as input_tokens,
        coalesce(output_tokens, 0) as output_tokens,
        (coalesce(input_tokens, 0) + coalesce(output_tokens, 0)) as total_tokens,
        coalesce(cost, 0.0) as token_cost_usd,
        coalesce(latency_ms, 0.0) as latency_ms,
        ttft_ms,
        
        -- Execution status
        coalesce(status, 'success') as execution_status,
        error_type,
        user_feedback,
        
        -- Automated evaluator metrics
        judge_score,
        judge_coherence,
        judge_relevance,
        case 
            when hallucination_flag = 1 or hallucination_flag = true then 1
            else 0
        end as is_hallucination
    from source
)

select * from renamed
