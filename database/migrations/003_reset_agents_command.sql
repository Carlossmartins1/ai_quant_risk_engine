-- =============================================================================
--  Migración 003 — Comando RESET_AGENTS en el plano de control
-- =============================================================================
--  Permite al dashboard pedir al orquestador que recapitalice a los agentes (colateral inicial,
--  cuenta ACTIVE, posiciones cerradas) para encadenar escenarios de Cisne Negro sin reiniciarlo.
--  El Insurance Fund no se toca.
--
--  Idempotente. ADD VALUE es válido dentro de una transacción desde PostgreSQL 12 (el valor no se
--  usa en esta misma transacción).
-- =============================================================================

ALTER TYPE simulation_command ADD VALUE IF NOT EXISTS 'RESET_AGENTS';
