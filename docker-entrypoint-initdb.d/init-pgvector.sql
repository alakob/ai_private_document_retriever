-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a dedicated user for the application with limited permissions
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user WITH LOGIN PASSWORD 'app_password';
    END IF;
END
$$;

-- Grant necessary permissions to app_user
GRANT CONNECT ON DATABASE "ragSystem" TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- Ensure future tables will also be accessible to app_user
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE ON SEQUENCES TO app_user;

-- Create a connection pool role to improve performance
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'connection_pool') THEN
        CREATE ROLE connection_pool WITH LOGIN PASSWORD 'pool_password';
    END IF;
END
$$;

-- Grant necessary permissions to connection_pool role
GRANT CONNECT ON DATABASE "ragSystem" TO connection_pool;
GRANT USAGE ON SCHEMA public TO connection_pool;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO connection_pool;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO connection_pool;

-- Set optimization parameters for PostgreSQL
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_cache_size = '512MB';
ALTER SYSTEM SET max_connections = 100;

-- Reload configuration
SELECT pg_reload_conf();
