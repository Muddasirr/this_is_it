-- Add dribble to Postgres eventtype enum (idempotent-ish)
DO $$ BEGIN
  ALTER TYPE eventtype ADD VALUE IF NOT EXISTS 'dribble';
EXCEPTION
  WHEN duplicate_object THEN NULL;
END $$;
