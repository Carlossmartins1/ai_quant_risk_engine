-- =============================================================================
--  Migración 002 — Plano de control de la simulación (Fase 6)
-- =============================================================================
--  Desacopla el dashboard (proceso Streamlit) del orquestador (otro proceso/contenedor):
--
--   * simulation_commands : cola append-only de órdenes del operador (auditable). El orquestador
--                           la consume con SELECT ... FOR UPDATE SKIP LOCKED (exactly-once aunque
--                           haya varias réplicas) y sella consumed_at / consumed_by.
--   * simulation_control  : fila única con el estado publicado por el orquestador: modo actual,
--                           heartbeat y snapshot de mercado (marks/oracle/liquidez) en JSONB, que
--                           el dashboard usa para calcular health ratios con los mismos precios
--                           que el motor de riesgo.
--
--  Idempotente: se aplica en instalaciones nuevas (vía schema.sql) y existentes (database.migrate).
-- =============================================================================

DO $$ BEGIN
    CREATE TYPE simulation_command AS ENUM ('SET_NORMAL', 'TRIGGER_BLACK_SWAN');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS simulation_commands (
    command_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    command        simulation_command NOT NULL,
    requested_by   TEXT        NOT NULL,
    requested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    consumed_at    TIMESTAMPTZ,
    consumed_by    TEXT,
    note           TEXT,
    CONSTRAINT chk_consumed_pair CHECK ((consumed_at IS NULL) = (consumed_by IS NULL))
);

-- Cola: solo las pendientes, en orden de llegada
CREATE INDEX IF NOT EXISTS idx_simulation_commands_pending
    ON simulation_commands (command_id)
    WHERE consumed_at IS NULL;

CREATE TABLE IF NOT EXISTS simulation_control (
    control_id       SMALLINT    PRIMARY KEY DEFAULT 1,
    current_mode     TEXT        NOT NULL DEFAULT 'OFFLINE',
    orchestrator_id  TEXT,
    heartbeat_at     TIMESTAMPTZ,
    market_state     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    stats            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_control_singleton CHECK (control_id = 1),
    CONSTRAINT chk_control_mode CHECK (current_mode IN ('OFFLINE', 'NORMAL', 'BLACK_SWAN'))
);

INSERT INTO simulation_control (control_id) VALUES (1) ON CONFLICT (control_id) DO NOTHING;

DROP TRIGGER IF EXISTS trg_simulation_control_touch ON simulation_control;
CREATE TRIGGER trg_simulation_control_touch
    BEFORE UPDATE ON simulation_control
    FOR EACH ROW EXECUTE FUNCTION fn_touch_updated_at();

-- Registro de migraciones aplicadas (lo gestiona database/migrate.py)
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT        PRIMARY KEY,
    checksum    TEXT        NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
