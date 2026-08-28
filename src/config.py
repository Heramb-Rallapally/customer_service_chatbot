"""Environment-backed project configuration without external client setup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Optional

from pydantic import BaseModel, SecretStr


class Settings(BaseModel):
    """Configuration values shared by future application components.

    All service-specific values are optional so importing and testing the shared
    foundation never requires OCI or Oracle Database credentials.
    """

    oci_config_profile: str = "DEFAULT"
    oci_compartment_id: Optional[str] = None
    oci_endpoint: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    llm_model: Optional[str] = None
    oracle_db_user: Optional[str] = None
    oracle_db_password: Optional[SecretStr] = None
    oracle_db_dsn: Optional[str] = None
    oracle_vs_table: Optional[str] = None
    oracle_conversation_table: Optional[str] = None
    api_base_url: str = "http://127.0.0.1:8000"
    api_auth_mode: str = "development"
    api_development_user_id: str = "local-demo-user"

    @classmethod
    def from_environment(
        cls, environ: Optional[Mapping[str, str]] = None
    ) -> "Settings":
        """Build settings from an environment mapping or the process environment."""

        values = os.environ if environ is None else environ

        def optional(name: str) -> Optional[str]:
            value = values.get(name)
            return value if value else None

        return cls(
            oci_config_profile=values.get("OCI_CONFIG_PROFILE") or "DEFAULT",
            oci_compartment_id=optional("OCI_COMPARTMENT_ID"),
            oci_endpoint=optional("OCI_ENDPOINT"),
            embedding_model=optional("EMBEDDING_MODEL"),
            embedding_dimension=(
                int(values["EMBEDDING_DIMENSION"])
                if values.get("EMBEDDING_DIMENSION")
                else None
            ),
            llm_model=optional("LLM_MODEL"),
            oracle_db_user=optional("ORACLE_DB_USER"),
            oracle_db_password=optional("ORACLE_DB_PASSWORD"),
            oracle_db_dsn=optional("ORACLE_DB_DSN"),
            oracle_vs_table=optional("ORACLEVS_TABLE"),
            oracle_conversation_table=optional("ORACLE_CONVERSATION_TABLE"),
            api_base_url=values.get("API_BASE_URL") or "http://127.0.0.1:8000",
            api_auth_mode=values.get("API_AUTH_MODE") or "development",
            api_development_user_id=(
                values.get("API_DEVELOPMENT_USER_ID") or "local-demo-user"
            ),
        )


def get_settings() -> Settings:
    """Return configuration from the current process environment."""

    return Settings.from_environment()
