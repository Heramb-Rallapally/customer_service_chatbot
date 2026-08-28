-- Provision this table before enabling ORACLE_CONVERSATION_TABLE.
-- The application intentionally does not run DDL at startup.
CREATE TABLE customer_chat_conversations (
    conversation_id VARCHAR2(128 CHAR) PRIMARY KEY,
    user_id         VARCHAR2(256 CHAR),
    state_json      CLOB NOT NULL CHECK (state_json IS JSON),
    summary         CLOB,
    version         NUMBER(19) NOT NULL,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL
);
