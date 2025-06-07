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

def insert_cycles_interpolated(cur, battery_id, cycles_data, file_size_mb=0):
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
            
            # Determine appropriate sample rate based on data size and file size
            # For very large datasets or files, use a much higher sample rate
            sample_rate = 10  # Default sample rate
            
            # Adjust based on data array length
            if data_length > 500000:
                sample_rate = 50
            elif data_length > 1000000:
                sample_rate = 100
            elif data_length > 4000000:
                sample_rate = 200
                
            # Further adjust based on file size if it's large
            if file_size_mb > 200:
                sample_rate = max(sample_rate * 2, 100)  # Double the sample rate for large files, minimum 1:100
            elif file_size_mb > 150:
                sample_rate = max(sample_rate * 1.5, 50)  # Increase by 50% for medium-large files
            
            logger.info(f"Using sample rate of 1:{sample_rate} for dataset with {data_length} points")
            
            # Safely get values from arrays with bounds checking
            def safe_get(key, index, default=None):
                if key in cycles_data and cycles_data[key] and index < len(cycles_data[key]):
                    return cycles_data[key][index]
                return default
            
            # Use batch inserts for better performance
            batch_size = 1000
            inserted_count = 0
            batch_data = []
            
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
                
                time = safe_get('time', i)
                voltage = safe_get('voltage', i)
                current = safe_get('current', i)
                temperature = safe_get('temperature', i)
                charge_capacity = safe_get('charge_capacity', i)
                discharge_capacity = safe_get('discharge_capacity', i)
                
                # Skip if all values are None
                if all(v is None for v in [time, voltage, current, temperature, charge_capacity, discharge_capacity]):
                    continue
                
                # Add to batch
                batch_data.append((battery_id, cycle_idx, time, voltage, current, temperature, charge_capacity, discharge_capacity))
                
                # Execute batch insert when batch is full
                if len(batch_data) >= batch_size:
                    try:
                        # Use executemany for batch insert
                        cur.executemany(
                            """
                            INSERT INTO battery_cycles_interpolated (
                                battery_id, cycle_index, time, voltage, current_, 
                                temperature, charge_capacity, discharge_capacity
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            batch_data
                        )
                        # Commit each batch immediately to save progress
                        conn = cur.connection
                        conn.commit()
                        inserted_count += len(batch_data)
                        logger.info(f"Inserted and committed batch of {len(batch_data)} points, total {inserted_count} so far")
                        batch_data = []  # Clear batch after insertion
                    except Exception as e:
                        logger.error(f"Error inserting batch: {e}")
                        # Continue with next batch even if this one fails
                        batch_data = []
            
            # Insert any remaining data points
            if batch_data:
                try:
                    cur.executemany(
                        """
                        INSERT INTO battery_cycles_interpolated (
                            battery_id, cycle_index, time, voltage, current_, 
                            temperature, charge_capacity, discharge_capacity
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        batch_data
                    )
                    # Commit the final batch immediately
                    conn = cur.connection
                    conn.commit()
                    inserted_count += len(batch_data)
                    logger.info(f"Inserted and committed final batch of {len(batch_data)} points, total {inserted_count}")
                except Exception as e:
                    logger.error(f"Error inserting final batch: {e}")

            
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
        # Check file size first
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        logger.info(f"Processing file: {file_path} (Size: {file_size_mb:.2f} MB)")
        
        # For very large files, use a more aggressive sampling rate
        if file_size_mb > 200:
            logger.info(f"Large file detected ({file_size_mb:.2f} MB). Using more aggressive sampling.")
        
        # Load JSON data with a memory-efficient approach for large files
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            return
        
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
        
        # Start a new transaction for battery and summary data
        conn.autocommit = False
        cur = conn.cursor()
        
        battery_id = insert_battery(cur, data)
        
        # Commit the battery insertion immediately
        conn.commit()
        logger.info(f"Committed battery data with ID {battery_id}")
        
        if 'summary' in data:
            insert_battery_summaries(cur, battery_id, data['summary'])
            # Commit the summary data immediately
            conn.commit()
            logger.info(f"Committed summary data for battery ID {battery_id}")
        
        # Insert cycles interpolated data if available
        # (This will handle its own commits in smaller batches)
        if 'cycles_interpolated' in data:
            insert_cycles_interpolated(cur, battery_id, data['cycles_interpolated'], file_size_mb)
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
            # First close the cursor if it's still open
            if cur is not None and not cur.closed:
                cur.close()
                cur = None
                
            # Then handle the connection - make sure we're not in a transaction
            if conn and not conn.closed:
                # If we're in a transaction, roll it back before setting autocommit
                if not conn.autocommit:
                    try:
                        conn.rollback()
                    except Exception as rollback_error:
                        logger.error(f"Error during final rollback: {rollback_error}")
                # Now it's safe to set autocommit
                conn.autocommit = True
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")


def ingest_data(folder_path=None):
    """Main function to ingest data from JSON files."""
    # Load environment variables first to ensure we have access to FOLDER_DATA
    load_dotenv('.env')
    
    # Get folder path from environment variable if not provided
    if folder_path is None:
        folder_path = os.getenv('FOLDER_DATA', '/home/rangdong/DataBattery')
    
    logger.info(f"Using data folder: {folder_path}")
    
    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Track processed files to avoid reprocessing on restart
    processed_files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_files.txt')
    processed_files = set()
    
    # Load previously processed files if the file exists
    if os.path.exists(processed_files_path):
        try:
            with open(processed_files_path, 'r') as f:
                processed_files = set(line.strip() for line in f if line.strip())
            logger.info(f"Loaded {len(processed_files)} previously processed files")
        except Exception as e:
            logger.error(f"Error loading processed files list: {e}")
    
    # Connect to the database
    conn = None
    try:
        conn = connect_to_db()
        
        # Process each file
        for file_path in json_files:
            # Skip already processed files
            file_basename = os.path.basename(file_path)
            if file_basename in processed_files:
                logger.info(f"Skipping already processed file: {file_basename}")
                continue
                
            try:
                process_file(conn, file_path)
                
                # Mark file as processed
                processed_files.add(file_basename)
                try:
                    with open(processed_files_path, 'a') as f:
                        f.write(file_basename + '\n')
                except Exception as e:
                    logger.error(f"Error updating processed files list: {e}")
                    
            except KeyboardInterrupt:
                logger.warning("Process interrupted by user. Saving progress...")
                break
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                # Continue with next file
            
        logger.info("Data ingestion process completed")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Error in data ingestion process: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ingest battery data from JSON files into PostgreSQL database')
    parser.add_argument('--folder', type=str, help='Path to folder containing JSON files')
    args = parser.parse_args()
    
    # Start data ingestion
    ingest_data(args.folder)
