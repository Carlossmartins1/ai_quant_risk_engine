-- =============================================================================
--  Migración 005 — Auto-Deleveraging (ADL)
-- =============================================================================
--  Cuando una liquidación deja el Insurance Fund en negativo, el motor ya no se limita a una alerta:
--  ejecuta un ADL pro-rata sobre las posiciones ganadoras (risk_engine/adl.py) en su propia
--  transacción ACID y lo deja auditado aquí.
--
--   * adl_events : una fila por ejecución de ADL (déficit, recuperado, residual, φ, estado).
--   * adl_fills  : una fila por posición desapalancada (cantidad cerrada, beneficio transferido).
--   * insurance_fund_ledger.entry_type += 'ADL_RECOVERY' (entrada al fondo).
--
--  Ambas tablas son append-only (mismo trigger que liquidation_events) y sus CHECK fijan las
--  identidades contables: recovered + residual = shortfall y balance_after = balance_before + recovered.
-- =============================================================================

ALTER TYPE insurance_entry_type ADD VALUE IF NOT EXISTS 'ADL_RECOVERY';

CREATE TABLE IF NOT EXISTS adl_events (
    adl_event_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    adl_batch_id         UUID            NOT NULL UNIQUE,
    trigger_batch_id     UUID,                                   -- liquidación que dejó el fondo en negativo
    status               TEXT            NOT NULL,               -- COVERED | PARTIAL | NO_COUNTERPARTIES
    shortfall            NUMERIC(36,18)  NOT NULL,               -- −balance del fondo antes del ADL
    recovered            NUMERIC(36,18)  NOT NULL,
    residual_shortfall   NUMERIC(36,18)  NOT NULL,
    phi                  NUMERIC(20,12)  NOT NULL,               -- fracción común de desapalancamiento
    counterparties       INTEGER         NOT NULL,               -- cuentas desapalancadas
    positions            INTEGER         NOT NULL,
    fund_balance_before  NUMERIC(36,18)  NOT NULL,
    fund_balance_after   NUMERIC(36,18)  NOT NULL,
    market_regime        market_regime   NOT NULL DEFAULT 'LIVE_TESTNET',
    metadata             JSONB           NOT NULL DEFAULT '{}'::jsonb,
    created_at           TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT chk_adl_status    CHECK (status IN ('COVERED', 'PARTIAL', 'NO_COUNTERPARTIES')),
    CONSTRAINT chk_adl_amounts   CHECK (shortfall > 0 AND recovered >= 0 AND residual_shortfall >= 0
                                        AND recovered + residual_shortfall = shortfall),
    CONSTRAINT chk_adl_phi       CHECK (phi >= 0 AND phi <= 1),
    CONSTRAINT chk_adl_balance   CHECK (fund_balance_after = fund_balance_before + recovered),
    CONSTRAINT chk_adl_counts    CHECK (counterparties >= 0 AND positions >= counterparties)
);

COMMENT ON TABLE adl_events IS
    'Auto-Deleveraging: socialización del bad debt que el Insurance Fund no cubre (append-only).';

CREATE TABLE IF NOT EXISTS adl_fills (
    adl_fill_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    adl_event_id           BIGINT          NOT NULL REFERENCES adl_events (adl_event_id),
    user_id                BIGINT          NOT NULL REFERENCES users_collateral (user_id),
    position_id            BIGINT          NOT NULL REFERENCES open_positions (position_id),
    symbol                 TEXT            NOT NULL,
    quantity_before        NUMERIC(36,18)  NOT NULL,
    quantity_closed        NUMERIC(36,18)  NOT NULL,             -- mismo signo que la posición
    entry_price            NUMERIC(36,18)  NOT NULL,
    mark_price             NUMERIC(36,18)  NOT NULL,
    unrealized_pnl_before  NUMERIC(36,18)  NOT NULL,
    haircut                NUMERIC(36,18)  NOT NULL,             -- beneficio transferido al fondo
    account_phi            NUMERIC(20,12)  NOT NULL,
    created_at             TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_adl_fill_position UNIQUE (adl_event_id, position_id),
    CONSTRAINT chk_adl_fill_side    CHECK (quantity_closed * quantity_before > 0
                                           AND abs(quantity_closed) <= abs(quantity_before)),
    CONSTRAINT chk_adl_fill_winner  CHECK (unrealized_pnl_before > 0 AND haircut >= 0
                                           AND haircut <= unrealized_pnl_before),
    CONSTRAINT chk_adl_fill_phi     CHECK (account_phi > 0 AND account_phi <= 1)
);

DROP TRIGGER IF EXISTS trg_adl_events_immutable ON adl_events;
CREATE TRIGGER trg_adl_events_immutable
    BEFORE UPDATE OR DELETE ON adl_events
    FOR EACH ROW EXECUTE FUNCTION fn_forbid_mutation();

DROP TRIGGER IF EXISTS trg_adl_fills_immutable ON adl_fills;
CREATE TRIGGER trg_adl_fills_immutable
    BEFORE UPDATE OR DELETE ON adl_fills
    FOR EACH ROW EXECUTE FUNCTION fn_forbid_mutation();

CREATE INDEX IF NOT EXISTS idx_adl_events_created_brin ON adl_events USING BRIN (created_at);
CREATE INDEX IF NOT EXISTS idx_adl_fills_event ON adl_fills (adl_event_id);
CREATE INDEX IF NOT EXISTS idx_adl_fills_user_time ON adl_fills (user_id, created_at DESC);
