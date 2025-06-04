CREATE TABLE batteries (
    id SERIAL PRIMARY KEY,
    barcode TEXT UNIQUE NOT NULL,
    protocol TEXT,
    channel_id INTEGER,
    cycle_life INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE battery_summaries (
    id SERIAL PRIMARY KEY,
    battery_id INTEGER REFERENCES batteries(id) ON DELETE CASCADE,
    cycle_index INTEGER,
    discharge_capacity FLOAT,
    charge_capacity FLOAT,
    discharge_energy FLOAT,
    charge_energy FLOAT,
    dc_internal_resistance FLOAT,
    temperature_maximum FLOAT,
    temperature_average FLOAT,
    temperature_minimum FLOAT,
    date_time_iso TIMESTAMPTZ,
    energy_efficiency FLOAT,
    charge_throughput FLOAT,
    energy_throughput FLOAT,
    charge_duration FLOAT,
    time_temperature_integrated FLOAT,
    paused INTEGER
);

CREATE TABLE battery_cycles_interpolated (
    id SERIAL PRIMARY KEY,
    battery_id INTEGER REFERENCES batteries(id) ON DELETE CASCADE,
    cycle_index INTEGER NOT NULL,
    time FLOAT,
    voltage FLOAT,
    current_ FLOAT,
    temperature FLOAT,
    charge_capacity FLOAT,
    discharge_capacity FLOAT
);