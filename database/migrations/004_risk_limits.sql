-- =============================================================================
--  Migración 004 — Límites de riesgo por cuenta (aplicados server-side por la pasarela)
-- =============================================================================
--  Incidente 2026-11-16: el market_manipulator llegó a ~$17M de nocional (≈15x efectivo) porque el
--  único freno era el guardarraíl del propio agente. Los límites pasan a ser un atributo de la cuenta
--  que el OrderGateway verifica bajo lock antes de aumentar exposición (defensa en profundidad):
--
--    * max_leverage            (ya existía, no se aplicaba): nocional bruto ≤ max_leverage × equity
--    * max_gross_notional_usd  (nuevo, NULL = sin tope):    nocional bruto ≤ tope absoluto en USD
--
--  Las órdenes que REDUCEN exposición nunca se bloquean por límites.
-- =============================================================================

ALTER TABLE users_collateral
    ADD COLUMN IF NOT EXISTS max_gross_notional_usd NUMERIC(36,18);

ALTER TABLE users_collateral DROP CONSTRAINT IF EXISTS chk_max_gross_notional;
ALTER TABLE users_collateral
    ADD CONSTRAINT chk_max_gross_notional
    CHECK (max_gross_notional_usd IS NULL OR max_gross_notional_usd > 0);

COMMENT ON COLUMN users_collateral.max_leverage IS
    'Apalancamiento bruto máximo de la cuenta (Σ|q|·mark / equity). Lo aplica OrderGateway.';
COMMENT ON COLUMN users_collateral.max_gross_notional_usd IS
    'Tope absoluto de nocional bruto en USD (NULL = sin tope). Lo aplica OrderGateway.';
