from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    env: str = "development"
    log_level: str = "INFO"

    database_url: str
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    GROQ_API_KEY: str = "GROQ_API_KEY"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
