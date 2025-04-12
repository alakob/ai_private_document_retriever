#!/bin/bash
set -e

echo "Setting up pgAdmin auto-connection to PostgreSQL..."

# Wait for PostgreSQL to be fully up
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Get the directory for the current user based on PGADMIN_DEFAULT_EMAIL
EMAIL_HASH=$(echo ${PGADMIN_DEFAULT_EMAIL:-admin@example.com} | sed 's/@/_/g')
STORAGE_DIR="/var/lib/pgadmin/storage/$EMAIL_HASH"

# Create directory if it doesn't exist
mkdir -p "$STORAGE_DIR"

# Create servers.json with environment variables
cat > "$STORAGE_DIR/servers.json" << EOL
{
    "Servers": {
        "1": {
            "Name": "RAG Postgres",
            "Group": "Servers",
            "Host": "postgres",
            "Port": 5432,
            "MaintenanceDB": "${POSTGRES_DB:-ragSystem}",
            "Username": "${POSTGRES_USER:-postgres}",
            "SSLMode": "prefer",
            "PassFile": "$STORAGE_DIR/pgpass",
            "SavePassword": true
        }
    }
}
EOL

# Create pgpass file for passwordless connection in the storage directory instead of root
cat > "$STORAGE_DIR/pgpass" << EOL
postgres:5432:${POSTGRES_DB:-ragSystem}:${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:-yourpassword}
EOL

# Ensure correct permissions
chmod 600 "$STORAGE_DIR/servers.json"
chmod 600 "$STORAGE_DIR/pgpass"

# Create pgadmin preferences with auto-save passwords
mkdir -p "$STORAGE_DIR/preferences"
cat > "$STORAGE_DIR/preferences/preferences.json" << EOL
{
    "Browser": {
        "LastDatabase": "${POSTGRES_DB:-ragSystem}",
        "LastSchema": "public",
        "LastServer": "RAG Postgres"
    },
    "Servers": {
        "1": {
            "ConnectionParameters": {
                "host": "postgres",
                "password": "${POSTGRES_PASSWORD:-yourpassword}",
                "port": "5432",
                "username": "${POSTGRES_USER:-postgres}"
            }
        }
    },
    "SQLEditor": {
        "LargeTableRowCount": 1000
    },
    "ShowServerMode": false,
    "ServerMode": false,
    "PasswordSaveEnabled": true
}
EOL

chmod 600 "$STORAGE_DIR/preferences/preferences.json"

echo "✅ PostgreSQL server configuration complete!"

# Start pgAdmin
exec /entrypoint.sh
