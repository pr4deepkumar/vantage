from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    or_,
    text,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

from vantage.core.settings import settings

DATABASE_URL = settings.database_url

_url = make_url(DATABASE_URL)
if _url.get_backend_name() == "sqlite" and _url.database not in (None, "", ":memory:"):
    Path(_url.database).parent.mkdir(parents=True, exist_ok=True)

# check_same_thread=False: requests and the background evaluator share the SQLite connection pool.
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Trace(Base):
    """One span. ``parent_id`` links spans into a trace tree; a span without a parent is a trace root."""

    __tablename__ = "traces"

    id = Column(String(36), primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)  # span start (UTC)
    prompt_version = Column(String(50), index=True)
    model = Column(String(50), index=True, nullable=True)
    parent_id = Column(String(36), index=True, nullable=True)
    span_kind = Column(String(20), index=True, default="llm")  # "llm", "agent", "chain" or "tool"
    input = Column(Text)
    output = Column(Text, nullable=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    # USD; empty when the model's price is unknown. No column default: the ORM would put it in place of None.
    cost = Column(Float, nullable=True)
    latency_ms = Column(Float, default=0.0)
    ttft_ms = Column(Float, nullable=True)  # time to first text token, streamed calls only
    status = Column(String(20), index=True)  # "success" or "error"
    error_type = Column(String(50), nullable=True)  # "timeout", "rate_limit", "malformed_json", ...
    stop_reason = Column(String(30), nullable=True)  # "end_turn", "max_tokens", "refusal", ...
    user_feedback = Column(String(10), nullable=True)  # "up" or "down"

    # Evaluation, filled in after the call returns
    judge_score = Column(Integer, nullable=True)  # 1-5 overall quality
    judge_coherence = Column(Integer, nullable=True)  # 1-5
    judge_relevance = Column(Integer, nullable=True)  # 1-5
    hallucination_flag = Column(Boolean, nullable=True)  # a claim the RAG context doesn't support
    rag_context = Column(Text, nullable=True)  # RAG context given to the model

    def to_dict(self):
        ts = self.timestamp
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)  # stored as UTC; say so, or browsers read it as local time
        return {column.name: getattr(self, column.name) for column in self.__table__.columns} | {
            "timestamp": ts.isoformat() if ts else None
        }


def init_db():
    Base.metadata.create_all(bind=engine)
    # Databases created by an earlier version lack the newer columns; add them in place.
    existing = {column["name"] for column in inspect(engine).get_columns(Trace.__tablename__)}
    with engine.begin() as conn:
        for column in Trace.__table__.columns:
            if column.name not in existing:
                column_type = column.type.compile(engine.dialect)
                conn.execute(
                    text(f"ALTER TABLE {Trace.__tablename__} ADD COLUMN {column.name} {column_type}")
                )


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
        ts = trace_data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now(timezone.utc)

        columns = {column.name for column in Trace.__table__.columns}
        trace = Trace(
            **{key: value for key, value in trace_data.items() if key in columns} | {"timestamp": ts}
        )
        session.add(trace)
        session.flush()  # populate column defaults
        return trace.to_dict()


def update_trace_feedback(trace_id: str, feedback: str) -> bool:
    with get_db() as session:
        trace = session.get(Trace, trace_id)
        if trace:
            trace.user_feedback = feedback
            return True
        return False


def update_trace_judgement(trace_id: str, **fields) -> None:
    """Store evaluation results (judge_* and hallucination_flag) for a span."""
    with get_db() as session:
        trace = session.get(Trace, trace_id)
        if trace:
            for key, value in fields.items():
                setattr(trace, key, value)


def has_traces() -> bool:
    with get_db() as session:
        return session.query(Trace.id).first() is not None


def query_traces(
    *,
    span_kind: str | None = None,
    status: str | None = None,
    prompt_version: str | None = None,
    model: str | None = None,
    search: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Spans matching every filter given, newest first."""
    with get_db() as session:
        query = session.query(Trace)
        for column, value in (
            (Trace.span_kind, span_kind),
            (Trace.status, status),
            (Trace.prompt_version, prompt_version),
            (Trace.model, model),
        ):
            if value:
                query = query.filter(column == value)
        if search:
            literal = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{literal}%"
            query = query.filter(
                or_(*(column.ilike(pattern, escape="\\") for column in (Trace.input, Trace.output, Trace.id)))
            )
        query = query.order_by(Trace.timestamp.desc())
        if limit:
            query = query.limit(limit)
        return [trace.to_dict() for trace in query.all()]


def get_trace_by_id(trace_id: str):
    with get_db() as session:
        trace = session.get(Trace, trace_id)
        return trace.to_dict() if trace else None


def get_children(parent_id: str) -> list[dict]:
    """Direct child spans, in start order."""
    with get_db() as session:
        children = session.query(Trace).filter(Trace.parent_id == parent_id).order_by(Trace.timestamp).all()
        return [child.to_dict() for child in children]
