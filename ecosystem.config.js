module.exports = {
  apps: [{
    name: "battery-data-ingestion-3",
    script: "main.py",
    interpreter: "python3",
    // No need to specify folder as it will be read from .env file
    watch: false,
    instances: 1,
    autorestart: true,
    max_memory_restart: "1G",
    env: {
      NODE_ENV: "development"
    },
    env_production: {
      NODE_ENV: "production"
    }
  }]
};
