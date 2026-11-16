-- =============================================================================
--  AI-Driven Quant Risk & Liquidation Engine — Esquema Fase 1 (PostgreSQL 15+)
-- =============================================================================
--  Convenciones de diseño
--  ----------------------
--  * Importes y precios en NUMERIC(36,18): aritmética decimal exacta. Nunca FLOAT
--    en un ledger (el cómputo vectorizado en NumPy ocurre fuera, en memoria).
--  * La cantidad de una posición es CON SIGNO (+long / -short). El netting
--    cross-margin se reduce a SUM(quantity) y el PnL a quantity * (mark - entry),
--    sin ramas por lado -> 100% vectorizable.
--  * Invariante de netting: como máximo UNA posición OPEN por (user_id, symbol),
--    garantizado por el índice único parcial idx_user_risk.
--  * Los registros de auditoría (liquidation_events, insurance_fund_ledger) son
--    append-only: UPDATE/DELETE bloqueados por trigger.
--  * Protocolo de bloqueo (anti-deadlock), SIEMPRE en este orden:
--        users_collateral -> open_positions (ORDER BY position_id) -> insurance_fund
--  * Script idempotente: puede re-ejecutarse sin error.
-- =============================================================================

BEGIN;

SET client_min_messages = WARNING;

-- -----------------------------------------------------------------------------
-- 1. Tipos enumerados
-- -----------------------------------------------------------------------------
DO $$ BEGIN
    CREATE TYPE agent_profile AS ENUM (
        'WHALE_AGGRESSIVE',      -- Agente 1: 20x, colateral bajo
        'ARBITRAGEUR_NEUTRAL',   -- Agente 2: market-neutral, 2x-3x
        'MARKET_MANIPULATOR',    -- Agente 3: shock exógeno de liquidez
        'SYSTEM'                 -- cuentas internas (backstop, market maker)
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE margin_mode AS ENUM ('CROSS', 'ISOLATED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE account_status AS ENUM (
        'ACTIVE',        -- opera normalmente
        'LIQUIDATING',   -- bloqueada por el motor, liquidación en curso
        'LIQUIDATED',    -- liquidada; sin posiciones abiertas
        'FROZEN'         -- congelada manualmente (riesgo / compliance)
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE position_status AS ENUM ('OPEN', 'CLOSED', 'LIQUIDATED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE market_regime AS ENUM ('LIVE_TESTNET', 'STRESS_MONTE_CARLO');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE insurance_entry_type AS ENUM (
        'LIQUIDATION_FEE',      -- entrada: penalización cobrada al liquidado
        'BANKRUPTCY_DEFICIT',   -- salida: cubre equity negativa (bad debt)
        'CAPITAL_INJECTION',    -- entrada: capitalización del fondo
        'ADJUSTMENT'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- -----------------------------------------------------------------------------
-- 2. Funciones de trigger
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION fn_touch_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END $$;

-- updated_at + contador de versión (control de concurrencia optimista / auditoría)
CREATE OR REPLACE FUNCTION fn_touch_versioned()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    NEW.version    := OLD.version + 1;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION fn_forbid_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'La tabla % es append-only: % no permitido', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'insufficient_privilege';
END $$;

-- -----------------------------------------------------------------------------
-- 3. users_collateral — una cuenta de margen por agente
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users_collateral (
    user_id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    agent_name            TEXT            NOT NULL UNIQUE,
    agent_profile         agent_profile   NOT NULL,
    margin_mode           margin_mode     NOT NULL DEFAULT 'CROSS',
    deposited_collateral  NUMERIC(36,18)  NOT NULL DEFAULT 0,
    max_leverage          NUMERIC(6,2)    NOT NULL DEFAULT 20,
    account_status        account_status  NOT NULL DEFAULT 'ACTIVE',
    hl_address            TEXT            UNIQUE,          -- wallet Hyperliquid Testnet
    version               BIGINT          NOT NULL DEFAULT 0,
    created_at            TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Sin saldos negativos: el bad debt se absorbe por el Insurance Fund, nunca por la cuenta
    CONSTRAINT chk_collateral_non_negative CHECK (deposited_collateral >= 0),
    CONSTRAINT chk_max_leverage            CHECK (max_leverage >= 1 AND max_leverage <= 100),
    CONSTRAINT chk_hl_address_format       CHECK (hl_address IS NULL OR hl_address ~ '^0x[0-9a-fA-F]{40}$')
);

COMMENT ON TABLE  users_collateral IS 'Cuenta de margen por agente. Equity = deposited_collateral + SUM(unrealized PnL).';
COMMENT ON COLUMN users_collateral.deposited_collateral IS 'Colateral depositado + PnL realizado (USDC).';
COMMENT ON COLUMN users_collateral.version IS 'Se incrementa en cada UPDATE (trigger).';

DROP TRIGGER IF EXISTS trg_users_collateral_touch ON users_collateral;
CREATE TRIGGER trg_users_collateral_touch
    BEFORE UPDATE ON users_collateral
    FOR EACH ROW EXECUTE FUNCTION fn_touch_versioned();

-- Escaneo del motor: cuentas candidatas a evaluación de margen
CREATE INDEX IF NOT EXISTS idx_users_collateral_status
    ON users_collateral (account_status)
    WHERE account_status IN ('ACTIVE', 'LIQUIDATING');

-- -----------------------------------------------------------------------------
-- 4. open_positions — posiciones perpetuas (netting por usuario y símbolo)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS open_positions (
    position_id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id                  BIGINT          NOT NULL
                             REFERENCES users_collateral (user_id) ON DELETE RESTRICT,
    symbol                   TEXT            NOT NULL,               -- 'BTC', 'ETH' (naming Hyperliquid)
    quantity                 NUMERIC(36,18)  NOT NULL,               -- +long / -short (unidades del subyacente)
    side                     TEXT GENERATED ALWAYS AS
                             (CASE WHEN quantity > 0 THEN 'LONG' ELSE 'SHORT' END) STORED,
    entry_price              NUMERIC(36,18)  NOT NULL,
    leverage                 NUMERIC(6,2)    NOT NULL,
    initial_margin_rate      NUMERIC(20,18) GENERATED ALWAYS AS (1 / leverage) STORED,
    maintenance_margin_rate  NUMERIC(20,18)  NOT NULL,               -- MMR del activo (p.ej. 0.025)
    last_mark_price          NUMERIC(36,18),                         -- último mark visto por el motor
    status                   position_status NOT NULL DEFAULT 'OPEN',
    opened_at                TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ     NOT NULL DEFAULT now(),
    closed_at                TIMESTAMPTZ,

    CONSTRAINT chk_symbol_format    CHECK (symbol ~ '^[A-Z0-9]{2,20}$'),
    CONSTRAINT chk_quantity_nonzero CHECK (quantity <> 0),
    CONSTRAINT chk_entry_price_pos  CHECK (entry_price > 0),
    CONSTRAINT chk_mark_price_pos   CHECK (last_mark_price IS NULL OR last_mark_price > 0),
    CONSTRAINT chk_leverage_range   CHECK (leverage >= 1 AND leverage <= 100),
    -- El margen de mantenimiento debe ser estrictamente menor que el inicial
    CONSTRAINT chk_mmr_range        CHECK (maintenance_margin_rate > 0
                                           AND maintenance_margin_rate < 1 / leverage),
    -- Coherencia del ciclo de vida
    CONSTRAINT chk_closed_lifecycle CHECK ((status = 'OPEN') = (closed_at IS NULL))
);

COMMENT ON TABLE  open_positions IS 'Posiciones perpetuas. Una única fila OPEN por (user_id, symbol): netting cross-margin.';
COMMENT ON COLUMN open_positions.quantity IS 'Cantidad con signo. uPnL = quantity * (mark_price - entry_price).';
COMMENT ON COLUMN open_positions.maintenance_margin_rate IS 'MMR. Maintenance Margin = |quantity| * mark_price * MMR.';

DROP TRIGGER IF EXISTS trg_open_positions_touch ON open_positions;
CREATE TRIGGER trg_open_positions_touch
    BEFORE UPDATE ON open_positions
    FOR EACH ROW EXECUTE FUNCTION fn_touch_updated_at();

-- Índice principal del motor de riesgo:
--   * UNIQUE parcial  -> impone el invariante de netting (1 posición OPEN por user/símbolo).
--   * INCLUDE         -> index-only scan: el motor obtiene todo lo necesario para
--                        uPnL y Maintenance Margin sin tocar el heap.
--   * WHERE OPEN      -> el índice solo contiene riesgo vivo (pequeño y caliente en caché).
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_risk
    ON open_positions (user_id, symbol)
    INCLUDE (quantity, entry_price, leverage, maintenance_margin_rate)
    WHERE status = 'OPEN';

-- Fan-out por tick de mercado: "¿qué cuentas están expuestas a BTC?"
CREATE INDEX IF NOT EXISTS idx_open_positions_symbol
    ON open_positions (symbol)
    INCLUDE (user_id, quantity)
    WHERE status = 'OPEN';

-- -----------------------------------------------------------------------------
-- 5. insurance_fund — fila única (saldo caliente, se bloquea con FOR UPDATE)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS insurance_fund (
    fund_id     SMALLINT        PRIMARY KEY DEFAULT 1,
    balance     NUMERIC(36,18)  NOT NULL DEFAULT 0,
    version     BIGINT          NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ     NOT NULL DEFAULT now(),
    CONSTRAINT chk_singleton CHECK (fund_id = 1)
);

COMMENT ON TABLE insurance_fund IS
    'Fondo de seguro. balance < 0 implica fondo agotado -> disparar ADL (auto-deleveraging).';

DROP TRIGGER IF EXISTS trg_insurance_fund_touch ON insurance_fund;
CREATE TRIGGER trg_insurance_fund_touch
    BEFORE UPDATE ON insurance_fund
    FOR EACH ROW EXECUTE FUNCTION fn_touch_versioned();

INSERT INTO insurance_fund (fund_id, balance) VALUES (1, 0)
ON CONFLICT (fund_id) DO NOTHING;

-- -----------------------------------------------------------------------------
-- 6. liquidation_events — auditoría inmutable (1 fila por posición liquidada)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS liquidation_events (
    event_id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    liquidation_batch_id       UUID            NOT NULL,   -- agrupa todas las posiciones de UNA liquidación de cuenta
    user_id                    BIGINT          NOT NULL REFERENCES users_collateral (user_id),
    position_id                BIGINT          NOT NULL REFERENCES open_positions (position_id),
    symbol                     TEXT            NOT NULL,
    quantity                   NUMERIC(36,18)  NOT NULL,   -- cantidad con signo liquidada
    entry_price                NUMERIC(36,18)  NOT NULL,

    -- Microestructura en el instante del trigger
    mark_price                 NUMERIC(36,18)  NOT NULL,
    oracle_price               NUMERIC(36,18),
    execution_price            NUMERIC(36,18)  NOT NULL,   -- precio efectivo tras slippage en el L2
    bankruptcy_price           NUMERIC(36,18)  NOT NULL,   -- precio al que la equity de la cuenta = 0
    slippage_bps               NUMERIC(12,4),

    -- Estado de la cuenta ANTES de liquidar (snapshot de riesgo)
    collateral_before          NUMERIC(36,18)  NOT NULL,
    account_equity_before      NUMERIC(36,18)  NOT NULL,
    maintenance_margin_before  NUMERIC(36,18)  NOT NULL,
    margin_ratio_before        NUMERIC(20,10),             -- equity / MM (< 1 => liquidable)

    -- Resultado económico
    realized_pnl               NUMERIC(36,18)  NOT NULL,
    liquidation_fee            NUMERIC(36,18)  NOT NULL DEFAULT 0,
    deficit                    NUMERIC(36,18)  NOT NULL DEFAULT 0,  -- bad debt cubierto por el Insurance Fund
    insurance_fund_delta       NUMERIC(36,18)  NOT NULL DEFAULT 0,  -- fee - deficit

    market_regime              market_regime   NOT NULL DEFAULT 'LIVE_TESTNET',
    trigger_reason             TEXT            NOT NULL DEFAULT 'EQUITY_BELOW_MAINTENANCE',
    metadata                   JSONB           NOT NULL DEFAULT '{}'::jsonb,
    created_at                 TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_liquidation_batch_position UNIQUE (liquidation_batch_id, position_id), -- idempotencia
    CONSTRAINT chk_liq_quantity_nonzero CHECK (quantity <> 0),
    CONSTRAINT chk_liq_prices_positive  CHECK (mark_price > 0 AND execution_price > 0 AND bankruptcy_price >= 0),
    CONSTRAINT chk_liq_fee_non_negative CHECK (liquidation_fee >= 0),
    CONSTRAINT chk_liq_deficit_non_neg  CHECK (deficit >= 0),
    CONSTRAINT chk_liq_fund_delta       CHECK (insurance_fund_delta = liquidation_fee - deficit)
);

COMMENT ON TABLE liquidation_events IS 'Registro inmutable de liquidaciones (append-only).';

DROP TRIGGER IF EXISTS trg_liquidation_events_immutable ON liquidation_events;
CREATE TRIGGER trg_liquidation_events_immutable
    BEFORE UPDATE OR DELETE ON liquidation_events
    FOR EACH ROW EXECUTE FUNCTION fn_forbid_mutation();

CREATE INDEX IF NOT EXISTS idx_liquidation_events_user_time
    ON liquidation_events (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_liquidation_events_batch
    ON liquidation_events (liquidation_batch_id);

-- BRIN: índice minúsculo para consultas por rango temporal en una tabla append-only
CREATE INDEX IF NOT EXISTS idx_liquidation_events_created_brin
    ON liquidation_events USING BRIN (created_at);

-- -----------------------------------------------------------------------------
-- 7. insurance_fund_ledger — libro mayor inmutable del fondo de seguro
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS insurance_fund_ledger (
    entry_id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    entry_type            insurance_entry_type NOT NULL,
    amount                NUMERIC(36,18)  NOT NULL,     -- con signo: + entrada / - salida
    balance_after         NUMERIC(36,18)  NOT NULL,
    liquidation_event_id  BIGINT          REFERENCES liquidation_events (event_id),
    note                  TEXT,
    created_at            TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT chk_ledger_amount_nonzero CHECK (amount <> 0)
);

DROP TRIGGER IF EXISTS trg_insurance_fund_ledger_immutable ON insurance_fund_ledger;
CREATE TRIGGER trg_insurance_fund_ledger_immutable
    BEFORE UPDATE OR DELETE ON insurance_fund_ledger
    FOR EACH ROW EXECUTE FUNCTION fn_forbid_mutation();

CREATE INDEX IF NOT EXISTS idx_insurance_ledger_event
    ON insurance_fund_ledger (liquidation_event_id)
    WHERE liquidation_event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_insurance_ledger_created_brin
    ON insurance_fund_ledger USING BRIN (created_at);

COMMIT;
