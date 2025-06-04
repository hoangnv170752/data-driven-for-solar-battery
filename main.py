import os
import glob
import json
import logging
import psycopg2
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("battery_data_ingestion")

def connect_to_db():
    """Establish a connection to the PostgreSQL database."""
    try:
        load_dotenv('.env')
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            dbname=os.getenv('DB_NAME')
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def insert_battery(cur, data):
    """Insert battery data into the batteries table and return the battery ID."""
    try:
        # Extract cycle_life from summary data if available
        cycle_life = None
        if 'summary' in data and 'cycle_index' in data['summary']:
            cycle_indices = data['summary']['cycle_index']
            if cycle_indices:
                cycle_life = max(cycle_indices)
        
        # Insert into batteries table
        cur.execute(
            "INSERT INTO batteries (barcode, protocol, channel_id, cycle_life) VALUES (%s, %s, %s, %s) RETURNING id",
            (data.get('barcode'), data.get('protocol'), data.get('channel_id'), cycle_life)
        )
        battery_id = cur.fetchone()[0]
        return battery_id
    except Exception as e:
        logger.error(f"Error inserting battery data: {e}")
        raise

def insert_battery_summaries(cur, battery_id, summary_data):
    """Insert battery summary data into the battery_summaries table."""
    try:
        if not summary_data or 'cycle_index' not in summary_data:
            logger.warning(f"No summary data found for battery ID {battery_id}")
            return
        
        # Get the number of cycles
        num_cycles = len(summary_data['cycle_index'])
        
        for i in range(num_cycles):
            # Extract values for each cycle
            cycle_index = summary_data['cycle_index'][i] if 'cycle_index' in summary_data and i < len(summary_data['cycle_index']) else None
            discharge_capacity = summary_data['discharge_capacity'][i] if 'discharge_capacity' in summary_data and i < len(summary_data['discharge_capacity']) else None
            charge_capacity = summary_data['charge_capacity'][i] if 'charge_capacity' in summary_data and i < len(summary_data['charge_capacity']) else None
            discharge_energy = summary_data['discharge_energy'][i] if 'discharge_energy' in summary_data and i < len(summary_data['discharge_energy']) else None
            charge_energy = summary_data['charge_energy'][i] if 'charge_energy' in summary_data and i < len(summary_data['charge_energy']) else None
            dc_internal_resistance = summary_data['dc_internal_resistance'][i] if 'dc_internal_resistance' in summary_data and i < len(summary_data['dc_internal_resistance']) else None
            temperature_maximum = summary_data['temperature_maximum'][i] if 'temperature_maximum' in summary_data and i < len(summary_data['temperature_maximum']) else None
            temperature_average = summary_data['temperature_average'][i] if 'temperature_average' in summary_data and i < len(summary_data['temperature_average']) else None
            temperature_minimum = summary_data['temperature_minimum'][i] if 'temperature_minimum' in summary_data and i < len(summary_data['temperature_minimum']) else None
            
            # Convert ISO datetime string to timestamp if available
            date_time_iso = None
            if 'date_time_iso' in summary_data and i < len(summary_data['date_time_iso']) and summary_data['date_time_iso'][i]:
                date_time_iso = summary_data['date_time_iso'][i]
            
            energy_efficiency = summary_data['energy_efficiency'][i] if 'energy_efficiency' in summary_data and i < len(summary_data['energy_efficiency']) else None
            charge_throughput = summary_data['charge_throughput'][i] if 'charge_throughput' in summary_data and i < len(summary_data['charge_throughput']) else None
            energy_throughput = summary_data['energy_throughput'][i] if 'energy_throughput' in summary_data and i < len(summary_data['energy_throughput']) else None
            charge_duration = summary_data['charge_duration'][i] if 'charge_duration' in summary_data and i < len(summary_data['charge_duration']) else None
            time_temperature_integrated = summary_data['time_temperature_integrated'][i] if 'time_temperature_integrated' in summary_data and i < len(summary_data['time_temperature_integrated']) else None
            paused = summary_data['paused'][i] if 'paused' in summary_data and i < len(summary_data['paused']) else None
            
            # Insert into battery_summaries table
            cur.execute(
                """
                INSERT INTO battery_summaries (
                    battery_id, cycle_index, discharge_capacity, charge_capacity, 
                    discharge_energy, charge_energy, dc_internal_resistance, 
                    temperature_maximum, temperature_average, temperature_minimum, 
                    date_time_iso, energy_efficiency, charge_throughput, 
                    energy_throughput, charge_duration, time_temperature_integrated, paused
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    battery_id, cycle_index, discharge_capacity, charge_capacity, 
                    discharge_energy, charge_energy, dc_internal_resistance, 
                    temperature_maximum, temperature_average, temperature_minimum, 
                    date_time_iso, energy_efficiency, charge_throughput, 
                    energy_throughput, charge_duration, time_temperature_integrated, paused
                )
            )
    except Exception as e:
        logger.error(f"Error inserting battery summary data: {e}")
        raise

def insert_cycles_interpolated(cur, battery_id, cycles_data):
    """Insert battery cycles interpolated data into the battery_cycles_interpolated table."""
    try:
        if not cycles_data:
            logger.warning(f"No cycles interpolated data found for battery ID {battery_id}")
            return
        
        logger.info(f"Cycles data keys for battery ID {battery_id}: {list(cycles_data.keys())[:10]}...")
        
        expected_keys = ['voltage', 'current', 'temperature', 'charge_capacity', 'discharge_capacity', 'cycle_index']
        flattened_format = False
        
        matching_keys = [key for key in expected_keys if key in cycles_data]
        if matching_keys and all(isinstance(cycles_data[key], list) for key in matching_keys):
            flattened_format = True
            logger.info(f"Detected flattened format with arrays at top level for battery ID {battery_id}")
        
        if flattened_format:
            data_length = 0
            for key in matching_keys:
                if cycles_data[key]:
                    data_length = len(cycles_data[key])
                    logger.info(f"Found {key} array with length {data_length} for battery ID {battery_id}")
                    break
            
            if data_length == 0:
                logger.warning(f"No valid data arrays found in cycles_interpolated for battery ID {battery_id}")
                return
            
            # Check if we have cycle_index array
            has_cycle_index = 'cycle_index' in cycles_data and cycles_data['cycle_index'] and len(cycles_data['cycle_index']) > 0
            if has_cycle_index:
                logger.info(f"Found cycle_index array with length {len(cycles_data['cycle_index'])}")
            else:
                logger.warning(f"No valid cycle_index array found, will use default cycle_index=0")
            
            # Process each data point
            # To avoid inserting too many rows, let's sample the data (e.g., every 10th point)
            sample_rate = 10
            inserted_count = 0
            for i in range(0, data_length, sample_rate):
                if i >= data_length:
                    break
                    
                # Get cycle_index from the array if available, otherwise use 0
                cycle_idx = 0  # Default value
                if has_cycle_index and i < len(cycles_data['cycle_index']):
                    try:
                        cycle_idx = int(cycles_data['cycle_index'][i])
                    except (ValueError, TypeError):
                        # If conversion fails, use default
                        pass
                
                # Safely get values from arrays with bounds checking
                def safe_get(key, index, default=None):
                    if key in cycles_data and cycles_data[key] and index < len(cycles_data[key]):
                        return cycles_data[key][index]
                    return default
                
                time = safe_get('time', i)
                voltage = safe_get('voltage', i)
                current = safe_get('current', i)
                temperature = safe_get('temperature', i)
                charge_capacity = safe_get('charge_capacity', i)
                discharge_capacity = safe_get('discharge_capacity', i)
                
                # Skip if all values are None
                if all(v is None for v in [time, voltage, current, temperature, charge_capacity, discharge_capacity]):
                    continue
                
                # Insert into battery_cycles_interpolated table
                try:
                    cur.execute(
                        """
                        INSERT INTO battery_cycles_interpolated (
                            battery_id, cycle_index, time, voltage, current_, 
                            temperature, charge_capacity, discharge_capacity
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            battery_id, cycle_idx, time, voltage, current, 
                            temperature, charge_capacity, discharge_capacity
                        )
                    )
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Error inserting cycle data point: {e}")
            
            logger.info(f"Inserted {inserted_count} data points for battery ID {battery_id} in flattened format")
        else:
            # Check if the keys are numeric (cycle indices)
            numeric_keys = []
            for key in cycles_data.keys():
                try:
                    numeric_keys.append(int(key))
                except (ValueError, TypeError):
                    pass
            
            if numeric_keys:
                logger.info(f"Found {len(numeric_keys)} numeric cycle keys for battery ID {battery_id}")
            else:
                logger.warning(f"No numeric cycle keys found in cycles_interpolated for battery ID {battery_id}")
                # Try to log a sample of the keys to understand the structure
                sample_keys = list(cycles_data.keys())[:5]
                logger.info(f"Sample keys: {sample_keys}")
                return
            
            # Process each cycle if data is organized by cycle index
            total_inserted = 0
            for cycle_index, cycle_data in cycles_data.items():
                # Skip if not a valid cycle index (could be metadata)
                try:
                    cycle_idx = int(cycle_index)
                except ValueError:
                    continue
                
                # Check if cycle_data is a dictionary
                if not isinstance(cycle_data, dict):
                    logger.warning(f"Cycle data for index {cycle_index} is not a dictionary: {type(cycle_data)}")
                    continue
                
                # Get the length of the data arrays
                if 'voltage' not in cycle_data or not cycle_data['voltage']:
                    logger.debug(f"No voltage data for cycle {cycle_index} in battery ID {battery_id}")
                    continue
                    
                data_length = len(cycle_data['voltage'])
                logger.debug(f"Cycle {cycle_index} has {data_length} data points for battery ID {battery_id}")
                
                # Process each data point in the cycle - sample to reduce volume
                sample_rate = 10  # Only insert every 10th point to reduce database size
                inserted_count = 0
                for i in range(0, data_length, sample_rate):
                    time = cycle_data.get('time', [None] * data_length)[i] if 'time' in cycle_data and i < len(cycle_data['time']) else None
                    voltage = cycle_data.get('voltage', [None] * data_length)[i] if 'voltage' in cycle_data and i < len(cycle_data['voltage']) else None
                    current = cycle_data.get('current', [None] * data_length)[i] if 'current' in cycle_data and i < len(cycle_data['current']) else None
                    temperature = cycle_data.get('temperature', [None] * data_length)[i] if 'temperature' in cycle_data and i < len(cycle_data['temperature']) else None
                    charge_capacity = cycle_data.get('charge_capacity', [None] * data_length)[i] if 'charge_capacity' in cycle_data and i < len(cycle_data['charge_capacity']) else None
                    discharge_capacity = cycle_data.get('discharge_capacity', [None] * data_length)[i] if 'discharge_capacity' in cycle_data and i < len(cycle_data['discharge_capacity']) else None
                    
                    # Insert into battery_cycles_interpolated table
                    try:
                        cur.execute(
                            """
                            INSERT INTO battery_cycles_interpolated (
                                battery_id, cycle_index, time, voltage, current_, 
                                temperature, charge_capacity, discharge_capacity
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                battery_id, cycle_idx, time, voltage, current, 
                                temperature, charge_capacity, discharge_capacity
                            )
                        )
                        inserted_count += 1
                    except Exception as e:
                        logger.error(f"Error inserting cycle data point for cycle {cycle_index}: {e}")
                
                total_inserted += inserted_count
                
            logger.info(f"Inserted total of {total_inserted} data points across all cycles for battery ID {battery_id}")
            
            if total_inserted == 0:
                logger.warning(f"Failed to insert any cycle data points for battery ID {battery_id}")
                # Log a sample of the data structure to help diagnose
                if numeric_keys:
                    sample_cycle = cycles_data.get(str(numeric_keys[0]))
                    if sample_cycle:
                        logger.info(f"Sample cycle keys: {list(sample_cycle.keys())[:5]}")
                        if 'voltage' in sample_cycle:
                            logger.info(f"Sample voltage array length: {len(sample_cycle['voltage'])}")
                            logger.info(f"Sample voltage data: {sample_cycle['voltage'][:5]}")

    except Exception as e:
        logger.error(f"Error inserting cycles interpolated data: {e}")
        raise

def process_file(conn, file_path):
    """Process a single JSON file and insert its data into the database."""
    # Create a new cursor for this file
    cur = None
    
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Always set autocommit to True before creating a cursor
        # This ensures we're not in a transaction when setting session parameters
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if battery with this barcode already exists
        barcode = data.get('barcode')
        if barcode:
            cur.execute("SELECT id FROM batteries WHERE barcode = %s", (barcode,))
            existing_battery = cur.fetchone()
            
            if existing_battery:
                logger.info(f"Battery with barcode {barcode} already exists, skipping insertion")
                return
        
        # Close the cursor before starting a new transaction
        cur.close()
        cur = None
        
        # Start a new transaction
        conn.autocommit = False
        cur = conn.cursor()
        
        # Insert battery data and get battery ID
        battery_id = insert_battery(cur, data)
        
        # Insert summary data if available
        if 'summary' in data:
            insert_battery_summaries(cur, battery_id, data['summary'])
        
        # Insert cycles interpolated data if available
        if 'cycles_interpolated' in data:
            insert_cycles_interpolated(cur, battery_id, data['cycles_interpolated'])
        
        # Commit transaction
        conn.commit()
        logger.info(f"Successfully ingested data from {file_path}")
        
    except Exception as e:
        # Rollback in case of error
        if conn and not conn.closed and not conn.autocommit:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")
        logger.error(f"Error processing file {file_path}: {e}")
    finally:
        # Reset autocommit and close cursor
        try:
            if conn and not conn.closed:
                conn.autocommit = True
            if cur is not None and not cur.closed:
                cur.close()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")


def ingest_data(folder_path=None):
    """Main function to ingest data from JSON files."""
    # Load environment variables first to ensure we have access to FOLDER_DATA
    load_dotenv('.env')
    
    if folder_path is None:
        folder_path = os.getenv('FOLDER_DATA', '/home/rangdong/DataBattery')
    
    logger.info(f"Using data folder: {folder_path}")
    
    conn = None
    try:
        # Connect to the database
        conn = connect_to_db()
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            logger.error(f"Folder path does not exist: {folder_path}")
            return
            
        # Get list of JSON files
        json_files = glob.glob(os.path.join(folder_path, '*.json'))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each file
        for file_path in json_files:
            process_file(conn, file_path)
            
    except Exception as e:
        logger.error(f"Error in data ingestion process: {e}")
    finally:
        # Close database connection
        if conn:
            conn.close()
        logger.info("Data ingestion process completed")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ingest battery data from JSON files into PostgreSQL database')
    parser.add_argument('--folder', type=str, help='Path to folder containing JSON files')
    args = parser.parse_args()
    
    # Start data ingestion
    ingest_data(args.folder)
