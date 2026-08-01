import os
from datetime import datetime, timezone
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# Ensure data directory exists
DATA_DIR = os.getenv("VANTAGE_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")))
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_DB_PATH = os.path.join(DATA_DIR, "observability.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Create engine with thread pool sharing configuration for SQLite
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionLocal)

Base = declarative_base()

class Trace(Base):
    __tablename__ = "traces"

    id = Column(String(36), primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    prompt_version = Column(String(50), index=True)
    model = Column(String(50), index=True, nullable=True)
    parent_id = Column(String(36), index=True, nullable=True)
    span_kind = Column(String(20), index=True, default="llm")
    input = Column(Text)
    output = Column(Text, nullable=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    ttft_ms = Column(Float, nullable=True)
    status = Column(String(20), index=True) # "success" or "error"
    error_type = Column(String(50), nullable=True) # "timeout", "rate_limit", "malformed_json", "empty_response", "other"
    user_feedback = Column(String(10), nullable=True) # "up" or "down"
    
    # Judge metrics
    judge_score = Column(Integer, nullable=True) # 1-5 quality score
    judge_coherence = Column(Integer, nullable=True) # 1-5 coherence
    judge_relevance = Column(Integer, nullable=True) # 1-5 relevance
    hallucination_flag = Column(Boolean, nullable=True) # True/False
    rag_context = Column(Text, nullable=True) # RAG context used

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "parent_id": self.parent_id,
            "span_kind": self.span_kind,
            "input": self.input,
            "output": self.output,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "status": self.status,
            "error_type": self.error_type,
            "user_feedback": self.user_feedback,
            "judge_score": self.judge_score,
            "judge_coherence": self.judge_coherence,
            "judge_relevance": self.judge_relevance,
            "hallucination_flag": self.hallucination_flag,
            "rag_context": self.rag_context
        }

def init_db():
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_db():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def save_trace(trace_data: dict) -> dict:
    with get_db() as session:
        # Convert string timestamp to datetime if necessary
        ts = trace_data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now(timezone.utc)

        trace = Trace(
            id=trace_data["id"],
            timestamp=ts,
            prompt_version=trace_data.get("prompt_version"),
            model=trace_data.get("model"),
            parent_id=trace_data.get("parent_id"),
            span_kind=trace_data.get("span_kind", "llm"),
            input=trace_data.get("input"),
            output=trace_data.get("output"),
            input_tokens=trace_data.get("input_tokens", 0),
            output_tokens=trace_data.get("output_tokens", 0),
            cost=trace_data.get("cost", 0.0),
            latency_ms=trace_data.get("latency_ms", 0.0),
            ttft_ms=trace_data.get("ttft_ms"),
            status=trace_data.get("status"),
            error_type=trace_data.get("error_type"),
            user_feedback=trace_data.get("user_feedback"),
            judge_score=trace_data.get("judge_score"),
            judge_coherence=trace_data.get("judge_coherence"),
            judge_relevance=trace_data.get("judge_relevance"),
            hallucination_flag=trace_data.get("hallucination_flag"),
            rag_context=trace_data.get("rag_context")
        )
        session.add(trace)
        # Flush to populate default database fields
        session.flush()
        return trace.to_dict()

def update_trace_feedback(trace_id: str, feedback: str) -> bool:
    with get_db() as session:
        trace = session.query(Trace).filter(Trace.id == trace_id).first()
        if trace:
            trace.user_feedback = feedback
            return True
        return False

def get_all_traces():
    with get_db() as session:
        # We query and return the list of Python dictionaries to avoid session attachment issues
        traces = session.query(Trace).order_by(Trace.timestamp.desc()).all()
        return [t.to_dict() for t in traces]

def get_trace_by_id(trace_id: str):
    with get_db() as session:
        trace = session.query(Trace).filter(Trace.id == trace_id).first()
        return trace.to_dict() if trace else None
