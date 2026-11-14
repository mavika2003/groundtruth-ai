from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://groundtruth:groundtruth@localhost:5432/groundtruth"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_provider: str = "openai"
    gemini_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    embedding_dimensions: int = 1536
    jwt_secret: str = ""
    admin_token: str = ""
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    public_demo_enabled: bool = False
    demo_daily_limit: int = 20
    otel_exporter_otlp_endpoint: str = ""
    input_cost_per_million: float = 0.15
    output_cost_per_million: float = 0.60


settings = Settings()
