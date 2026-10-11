from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, ForeignKey
from sqlalchemy.orm import relationship
try:
    from .database import Base
except ImportError:
    from database import Base

def utcnow() -> datetime:
    """Naive UTC timestamp (datetime.utcnow() is deprecated since Python 3.12)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RequestLog(Base):
    __tablename__ = "request_logs"
    __table_args__ = {'extend_existing': True}

    request_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    model = Column(String)
    timestamp = Column(DateTime, default=utcnow)
    latency_ms = Column(Float)
    status = Column(String)
    
    content_log = relationship("ContentLog", back_populates="request_log", uselist=False)
    metrics_log = relationship("MetricsLog", back_populates="request_log", uselist=False)
    tool_logs = relationship("ToolExecutionLog", back_populates="request_log")

class ContentLog(Base):
    __tablename__ = "content_logs"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, ForeignKey("request_logs.request_id"), unique=True)
    user_query = Column(Text)
    bot_response = Column(Text)

    request_log = relationship("RequestLog", back_populates="content_log")

class MetricsLog(Base):
    __tablename__ = "metrics_logs"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, ForeignKey("request_logs.request_id"), unique=True)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_tokens = Column(Integer)
    input_cost = Column(Float)
    output_cost = Column(Float)
    total_cost = Column(Float)
    cost_per_token = Column(Float)
    input_output_ratio = Column(Float)
    model_used = Column(String)

    request_log = relationship("RequestLog", back_populates="metrics_log")


class DailyStats(Base):
    __tablename__ = "daily_stats"
    __table_args__ = {'extend_existing': True}

    date = Column(Date, primary_key=True)
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    success_count = Column(Integer, default=0)
    blocked_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)


class ToolExecutionLog(Base):
    """
    Logs tool-level security validations, authorization decisions,
    execution latency, timeouts, and token metrics.
    """
    __tablename__ = "tool_execution_logs"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, ForeignKey("request_logs.request_id"), index=True)
    tool_name = Column(String, nullable=False, index=True)
    tool_args = Column(Text)
    authorized = Column(String)  # 'APPROVED', 'BLOCKED', 'REQUIRES_HITL'
    block_reason = Column(String, nullable=True)
    execution_status = Column(String)  # 'SUCCESS', 'TIMEOUT', 'FAILED', 'BLOCKED'
    latency_ms = Column(Float, default=0.0)
    estimated_tokens = Column(Integer, default=0)
    output_snippet = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utcnow)

    request_log = relationship("RequestLog", back_populates="tool_logs")


class ApiKeyConfig(Base):
    """
    Stores virtual firewall API keys generated for users and applications,
    mapping them to a specific model and firewall policy.
    """
    __tablename__ = "api_key_configs"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    key_name = Column(String, nullable=False)
    api_key = Column(String, unique=True, index=True, nullable=False)
    model = Column(String, nullable=False)
    provider = Column(String, default="groq")
    created_at = Column(DateTime, default=utcnow)
    is_active = Column(Integer, default=1)
    prompt_injection_enabled = Column(Integer, default=1)
    tool_guard_enabled = Column(Integer, default=1)
    kill_switch_enabled = Column(Integer, default=1)
    sensitive_data_enabled = Column(Integer, default=1)  # PII, Secrets & Leakage filter
    gdpr_mode = Column(String, default="tokenize")       # 'tokenize' (reversible vault), 'mask', 'block'
    rate_limit_rpm = Column(Integer, default=60)
    total_calls = Column(Integer, default=0)

