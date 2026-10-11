from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class RequestLog(Base):
    __tablename__ = "request_logs"
    __table_args__ = {'extend_existing': True}

    request_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    model = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latency_ms = Column(Float)
    status = Column(String)
    
    content_log = relationship("ContentLog", back_populates="request_log", uselist=False)
    metrics_log = relationship("MetricsLog", back_populates="request_log", uselist=False)

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
