from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "aismarthome")
DB_USER = os.getenv("DB_USER", "datnd")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456a@")
DB_PORT = os.getenv("DB_PORT", "5432")

# Print connection parameters for debugging (hide password)
print(f"Database connection parameters:")
print(f"  Host: {DB_HOST}")
print(f"  Database: {DB_NAME}")
print(f"  User: {DB_USER}")
print(f"  Port: {DB_PORT}")

# Use TCP/IP connection instead of Unix socket with explicit host format
# Force TCP/IP by using explicit host+port format
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=prefer&application_name=solar_battery_api"

# Create SQLAlchemy engine and session
try:
    print(f"Attempting to connect to: {DATABASE_URL.replace(DB_PASSWORD, '***')}")
    
    # Force TCP/IP connection by using create_engine with explicit connect_args
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "connect_timeout": 10,
            "application_name": "solar_battery_api",
            "host": DB_HOST,  # Explicitly set host
            "port": DB_PORT   # Explicitly set port
        },
        pool_pre_ping=True,
        pool_recycle=300
    )
    
    # Test the connection
    with engine.connect() as connection:
        result = connection.execute("SELECT 1")
        print("SQLAlchemy connection test successful")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    using_postgres = True
    print("Successfully connected to PostgreSQL database via SQLAlchemy")
except Exception as e:
    print(f"Error connecting to database via SQLAlchemy: {e}")
    print("Will fall back to direct psycopg2 connections for queries")
    using_postgres = False
    # Still create an engine for SQLAlchemy model definitions
    engine = create_engine("postgresql://user:pass@localhost:5432/dummy", echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class Battery(Base):
    __tablename__ = "batteries"
    
    id = Column(Integer, primary_key=True, index=True)
    barcode = Column(Text, unique=True, nullable=False)
    protocol = Column(Text)
    channel_id = Column(Integer)
    cycle_life = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    summaries = relationship("BatterySummary", back_populates="battery", cascade="all, delete")
    cycles = relationship("BatteryCycleInterpolated", back_populates="battery", cascade="all, delete")

class BatterySummary(Base):
    __tablename__ = "battery_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    battery_id = Column(Integer, ForeignKey("batteries.id", ondelete="CASCADE"))
    cycle_index = Column(Integer)
    discharge_capacity = Column(Float)
    charge_capacity = Column(Float)
    discharge_energy = Column(Float)
    charge_energy = Column(Float)
    dc_internal_resistance = Column(Float)
    temperature_maximum = Column(Float)
    temperature_average = Column(Float)
    temperature_minimum = Column(Float)
    date_time_iso = Column(DateTime)
    energy_efficiency = Column(Float)
    charge_throughput = Column(Float)
    energy_throughput = Column(Float)
    charge_duration = Column(Float)
    time_temperature_integrated = Column(Float)
    paused = Column(Integer)
    
    # Relationship
    battery = relationship("Battery", back_populates="summaries")

class BatteryCycleInterpolated(Base):
    __tablename__ = "battery_cycles_interpolated"
    
    id = Column(Integer, primary_key=True, index=True)
    battery_id = Column(Integer, ForeignKey("batteries.id", ondelete="CASCADE"))
    cycle_index = Column(Integer, nullable=False)
    time = Column(Float)
    voltage = Column(Float)
    current_ = Column(Float)
    temperature = Column(Float)
    charge_capacity = Column(Float)
    discharge_capacity = Column(Float)
    
    # Relationship
    battery = relationship("Battery", back_populates="cycles")

# Pydantic models for request/response
class BatteryBase(BaseModel):
    barcode: str
    protocol: Optional[str] = None
    channel_id: Optional[int] = None
    cycle_life: Optional[int] = None

class BatteryCreate(BatteryBase):
    pass

class BatteryResponse(BatteryBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class BatterySummaryBase(BaseModel):
    cycle_index: int
    discharge_capacity: Optional[float] = None
    charge_capacity: Optional[float] = None
    discharge_energy: Optional[float] = None
    charge_energy: Optional[float] = None
    dc_internal_resistance: Optional[float] = None
    temperature_maximum: Optional[float] = None
    temperature_average: Optional[float] = None
    temperature_minimum: Optional[float] = None
    date_time_iso: Optional[datetime] = None
    energy_efficiency: Optional[float] = None
    charge_throughput: Optional[float] = None
    energy_throughput: Optional[float] = None
    charge_duration: Optional[float] = None
    time_temperature_integrated: Optional[float] = None
    paused: Optional[int] = None

class BatterySummaryCreate(BatterySummaryBase):
    battery_id: int

class BatterySummaryResponse(BatterySummaryBase):
    id: int
    battery_id: int
    
    class Config:
        orm_mode = True

class BatteryCycleBase(BaseModel):
    cycle_index: int
    time: Optional[float] = None
    voltage: Optional[float] = None
    current_: Optional[float] = None
    temperature: Optional[float] = None
    charge_capacity: Optional[float] = None
    discharge_capacity: Optional[float] = None

class BatteryCycleCreate(BatteryCycleBase):
    battery_id: int

class BatteryCycleResponse(BatteryCycleBase):
    id: int
    battery_id: int
    
    class Config:
        orm_mode = True

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI app
app = FastAPI(title="Solar Battery API", description="API for solar battery data management")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Battery endpoints - GET methods only
@app.get("/batteries/", response_model=List[BatteryResponse])
def read_batteries(
    skip: int = 0, 
    limit: int = 100, 
    barcode: Optional[str] = None,
    db: Session = Depends(get_db)
):
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            query = db.query(Battery)
            if barcode:
                query = query.filter(Battery.barcode == barcode)
            return query.offset(skip).limit(limit).all()
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    query = "SELECT * FROM batteries"
    params = []
    
    if barcode:
        query += " WHERE barcode = %s"
        params.append(barcode)
    
    query += f" ORDER BY id LIMIT {limit} OFFSET {skip}"
    
    results = execute_query(query, params)
    if not results:
        return []
    
    # Convert SQL results to Pydantic models
    return [BatteryResponse(**row) for row in results]

@app.get("/batteries/{barcode}", response_model=BatteryResponse)
def read_battery(barcode: str, db: Session = Depends(get_db)):
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            battery = db.query(Battery).filter(Battery.barcode == barcode).first()
            if battery is not None:
                return battery
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    query = "SELECT * FROM batteries WHERE barcode = %s"
    results = execute_query(query, [barcode])
    
    if not results:
        raise HTTPException(status_code=404, detail="Battery not found")
    
    return BatteryResponse(**results[0])

# Battery Summary endpoints - GET methods only
@app.get("/battery-summaries/", response_model=List[BatterySummaryResponse])
def read_battery_summaries(
    skip: int = 0, 
    limit: int = 100, 
    barcode: Optional[str] = None,
    db: Session = Depends(get_db)
):
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            if barcode:
                # Join with batteries table to filter by barcode
                battery = db.query(Battery).filter(Battery.barcode == barcode).first()
                if battery:
                    query = db.query(BatterySummary).filter(BatterySummary.battery_id == battery.id)
                else:
                    return []
            else:
                query = db.query(BatterySummary)
            return query.offset(skip).limit(limit).all()
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    if barcode:
        query = """
        SELECT bs.* 
        FROM battery_summaries bs
        JOIN batteries b ON bs.battery_id = b.id
        WHERE b.barcode = %s
        ORDER BY bs.id
        LIMIT %s OFFSET %s
        """
        params = [barcode, limit, skip]
    else:
        query = f"SELECT * FROM battery_summaries ORDER BY id LIMIT {limit} OFFSET {skip}"
        params = []
    
    results = execute_query(query, params)
    if not results:
        return []
    
    # Convert SQL results to Pydantic models
    return [BatterySummaryResponse(**row) for row in results]

@app.get("/battery-summaries/{barcode}/{cycle_index}", response_model=BatterySummaryResponse)
def read_battery_summary(barcode: str, cycle_index: int, db: Session = Depends(get_db)):
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            # Find battery by barcode first
            battery = db.query(Battery).filter(Battery.barcode == barcode).first()
            if not battery:
                raise HTTPException(status_code=404, detail="Battery not found")
                
            # Then find the summary with matching battery_id and cycle_index
            summary = db.query(BatterySummary).filter(
                BatterySummary.battery_id == battery.id,
                BatterySummary.cycle_index == cycle_index
            ).first()
            
            if summary is not None:
                return summary
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    query = """
    SELECT bs.* 
    FROM battery_summaries bs
    JOIN batteries b ON bs.battery_id = b.id
    WHERE b.barcode = %s AND bs.cycle_index = %s
    """
    results = execute_query(query, [barcode, cycle_index])
    
    if not results:
        raise HTTPException(status_code=404, detail="Battery summary not found")
    
    return BatterySummaryResponse(**results[0])

# Battery Cycle endpoints - GET methods only
@app.get("/battery-cycles/", response_model=List[BatteryCycleResponse])
def read_battery_cycles(
    skip: int = 0, 
    limit: int = 100, 
    barcode: Optional[str] = None,
    cycle_index: Optional[int] = None,
    db: Session = Depends(get_db)
):
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            if barcode:
                # Find battery by barcode first
                battery = db.query(Battery).filter(Battery.barcode == barcode).first()
                if not battery:
                    return []
                    
                # Then query cycles with matching battery_id
                query = db.query(BatteryCycleInterpolated).filter(
                    BatteryCycleInterpolated.battery_id == battery.id
                )
            else:
                query = db.query(BatteryCycleInterpolated)
                
            if cycle_index:
                query = query.filter(BatteryCycleInterpolated.cycle_index == cycle_index)
                
            return query.offset(skip).limit(limit).all()
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    conditions = []
    params = []
    
    if barcode:
        # Need to join with batteries table to filter by barcode
        query = """
        SELECT bci.* 
        FROM battery_cycles_interpolated bci
        JOIN batteries b ON bci.battery_id = b.id
        WHERE b.barcode = %s
        """
        params.append(barcode)
        
        if cycle_index:
            query += " AND bci.cycle_index = %s"
            params.append(cycle_index)
            
        query += f" ORDER BY bci.time LIMIT {limit} OFFSET {skip}"
    else:
        query = "SELECT * FROM battery_cycles_interpolated"
        
        if cycle_index:
            query += " WHERE cycle_index = %s"
            params.append(cycle_index)
            
        query += f" ORDER BY time LIMIT {limit} OFFSET {skip}"
    
    results = execute_query(query, params)
    if not results:
        return []
    
    # Convert SQL results to Pydantic models
    return [BatteryCycleResponse(**row) for row in results]

@app.get("/battery-cycles/{barcode}/{cycle_index}/{time_point}", response_model=BatteryCycleResponse)
def read_battery_cycle(barcode: str, cycle_index: int, time_point: float, db: Session = Depends(get_db)):
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            # Find battery by barcode first
            battery = db.query(Battery).filter(Battery.barcode == barcode).first()
            if not battery:
                raise HTTPException(status_code=404, detail="Battery not found")
                
            # Then find the cycle with matching battery_id, cycle_index and closest time point
            cycle = db.query(BatteryCycleInterpolated).filter(
                BatteryCycleInterpolated.battery_id == battery.id,
                BatteryCycleInterpolated.cycle_index == cycle_index,
                BatteryCycleInterpolated.time == time_point
            ).first()
            
            # If exact time point not found, get the closest one
            if not cycle:
                # Get closest time point
                closest_time = db.query(
                    BatteryCycleInterpolated.time
                ).filter(
                    BatteryCycleInterpolated.battery_id == battery.id,
                    BatteryCycleInterpolated.cycle_index == cycle_index
                ).order_by(
                    func.abs(BatteryCycleInterpolated.time - time_point)
                ).first()
                
                if closest_time:
                    cycle = db.query(BatteryCycleInterpolated).filter(
                        BatteryCycleInterpolated.battery_id == battery.id,
                        BatteryCycleInterpolated.cycle_index == cycle_index,
                        BatteryCycleInterpolated.time == closest_time[0]
                    ).first()
            
            if cycle is not None:
                return cycle
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    # First get the battery ID from barcode
    battery_query = "SELECT id FROM batteries WHERE barcode = %s"
    battery_results = execute_query(battery_query, [barcode])
    
    if not battery_results:
        raise HTTPException(status_code=404, detail="Battery not found")
    
    battery_id = battery_results[0]["id"]
    
    # Get the exact time point or closest one
    query = f"""
    SELECT * FROM battery_cycles_interpolated 
    WHERE battery_id = %s AND cycle_index = %s AND time = %s
    """
    results = execute_query(query, [battery_id, cycle_index, time_point])
    
    # If exact time point not found, get the closest one
    if not results:
        closest_time_query = f"""
        SELECT time FROM battery_cycles_interpolated 
        WHERE battery_id = %s AND cycle_index = %s
        ORDER BY ABS(time - %s) LIMIT 1
        """
        closest_time_results = execute_query(closest_time_query, [battery_id, cycle_index, time_point])
        
        if not closest_time_results:
            raise HTTPException(status_code=404, detail="Battery cycle not found")
        
        closest_time = closest_time_results[0]["time"]
        
        query = f"""
        SELECT * FROM battery_cycles_interpolated 
        WHERE battery_id = %s AND cycle_index = %s AND time = %s
        """
        results = execute_query(query, [battery_id, cycle_index, closest_time])
    
    if not results:
        raise HTTPException(status_code=404, detail="Battery cycle not found")
    
    return BatteryCycleResponse(**results[0])

# Direct PostgreSQL connection functions
def get_db_connection():
    """Create a direct connection to PostgreSQL database"""
    try:
        # Print connection attempt details
        print(f"Attempting direct psycopg2 connection to PostgreSQL:")
        print(f"  Host: {DB_HOST}")
        print(f"  Database: {DB_NAME}")
        print(f"  User: {DB_USER}")
        print(f"  Port: {DB_PORT}")
        
        # Create connection
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=10,
            application_name="solar_battery_api",
            sslmode="prefer"
        )
        
        # Test connection
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            print("Direct psycopg2 connection test successful")
            
        return conn
    except Exception as e:
        print(f"Error connecting directly to PostgreSQL: {e}")
        return None

def execute_query(query: str, params=None):
    """Execute a query and return results as dictionaries"""
    conn = get_db_connection()
    if not conn:
        print("No PostgreSQL connection available for direct query")
        return []
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            print(f"Executing SQL: {query}")
            print(f"With params: {params}")
            cur.execute(query, params)
            results = cur.fetchall()
            print(f"Query returned {len(results)} results")
            return [dict(row) for row in results]
    except Exception as e:
        print(f"Error executing query: {e}")
        return []
    finally:
        conn.close()

# Advanced queries are now integrated into the main endpoints
# The direct SQL functionality is now used as a fallback mechanism
# when SQLAlchemy connection fails

# Define a response model for battery statistics
class BatteryStatsResponse(BaseModel):
    id: int
    barcode: str
    protocol: str
    total_cycles: int
    avg_discharge_capacity: Optional[float] = None
    avg_charge_capacity: Optional[float] = None
    max_temperature: Optional[float] = None
    min_temperature: Optional[float] = None
    
    class Config:
        orm_mode = True

# Battery statistics endpoint
@app.get("/battery-stats/", response_model=List[BatteryStatsResponse])
def get_battery_stats(barcode: Optional[str] = None):
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            # Use SQLAlchemy to query statistics
            with engine.connect() as connection:
                base_query = """
                SELECT 
                    b.id, 
                    b.barcode, 
                    b.protocol, 
                    COUNT(DISTINCT bs.cycle_index) as total_cycles,
                    AVG(bs.discharge_capacity) as avg_discharge_capacity,
                    AVG(bs.charge_capacity) as avg_charge_capacity,
                    MAX(bs.temperature_maximum) as max_temperature,
                    MIN(bs.temperature_minimum) as min_temperature
                FROM 
                    batteries b
                LEFT JOIN 
                    battery_summaries bs ON b.id = bs.battery_id
                """
                
                if barcode:
                    base_query += " WHERE b.barcode = :barcode"
                    params = {"barcode": barcode}
                else:
                    params = {}
                    
                base_query += """
                GROUP BY 
                    b.id, b.barcode, b.protocol
                ORDER BY 
                    b.id
                """
                
                result = connection.execute(text(base_query), params)
                return [dict(row) for row in result]
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    base_query = """
    SELECT 
        b.id, 
        b.barcode, 
        b.protocol, 
        COUNT(DISTINCT bs.cycle_index) as total_cycles,
        AVG(bs.discharge_capacity) as avg_discharge_capacity,
        AVG(bs.charge_capacity) as avg_charge_capacity,
        MAX(bs.temperature_maximum) as max_temperature,
        MIN(bs.temperature_minimum) as min_temperature
    FROM 
        batteries b
    LEFT JOIN 
        battery_summaries bs ON b.id = bs.battery_id
    """
    
    params = []
    if barcode:
        base_query += " WHERE b.barcode = %s"
        params.append(barcode)
        
    base_query += """
    GROUP BY 
        b.id, b.barcode, b.protocol
    ORDER BY 
        b.id
    """
    
    results = execute_query(base_query, params)
    if not results:
        return []
    
    return [BatteryStatsResponse(**row) for row in results]

# Define a response model for battery cycle details
class BatteryCycleDetailResponse(BaseModel):
    battery_id: int
    cycle_index: int
    time: float
    voltage: float
    current_: float
    temperature: float
    charge_capacity: float
    discharge_capacity: float
    discharge_energy: Optional[float] = None
    charge_energy: Optional[float] = None
    energy_efficiency: Optional[float] = None
    
    class Config:
        orm_mode = True

# Battery cycle details endpoint
@app.get("/battery-cycle-details/{barcode}", response_model=List[BatteryCycleDetailResponse])
def get_battery_cycle_details(barcode: str, cycle_index: Optional[int] = None):
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            # Find battery by barcode first
            battery = db.query(Battery).filter(Battery.barcode == barcode).first()
            if not battery:
                return []
                
            # Use SQLAlchemy to query cycle details
            with engine.connect() as connection:
                query = text("""
                SELECT 
                    bci.battery_id,
                    bci.cycle_index,
                    bci.time,
                    bci.voltage,
                    bci.current_,
                    bci.temperature,
                    bci.charge_capacity,
                    bci.discharge_capacity,
                    bs.discharge_energy,
                    bs.charge_energy,
                    bs.energy_efficiency
                FROM 
                    battery_cycles_interpolated bci
                JOIN 
                    battery_summaries bs ON bci.battery_id = bs.battery_id AND bci.cycle_index = bs.cycle_index
                JOIN
                    batteries b ON bci.battery_id = b.id
                WHERE 
                    b.barcode = :barcode
                """)
                
                params = {"barcode": barcode}
                
                if cycle_index:
                    query = text(query.text + " AND bci.cycle_index = :cycle_index")
                    params["cycle_index"] = cycle_index
                
                query = text(query.text + " ORDER BY bci.cycle_index, bci.time LIMIT 1000")
                
                result = connection.execute(query, params)
                return [dict(row) for row in result]
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    query = """
    SELECT 
        bci.battery_id,
        bci.cycle_index,
        bci.time,
        bci.voltage,
        bci.current_,
        bci.temperature,
        bci.charge_capacity,
        bci.discharge_capacity,
        bs.discharge_energy,
        bs.charge_energy,
        bs.energy_efficiency
    FROM 
        battery_cycles_interpolated bci
    JOIN 
        battery_summaries bs ON bci.battery_id = bs.battery_id AND bci.cycle_index = bs.cycle_index
    JOIN
        batteries b ON bci.battery_id = b.id
    WHERE 
        b.barcode = %s
    """
    
    params = [barcode]
    
    if cycle_index:
        query += " AND bci.cycle_index = %s"
        params.append(cycle_index)
    
    query += " ORDER BY bci.cycle_index, bci.time LIMIT 1000"
    
    results = execute_query(query, params)
    if not results:
        return []
    
    return [BatteryCycleDetailResponse(**row) for row in results]

# Database schema information endpoints
@app.get("/tables")
def list_tables():
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            # Use SQLAlchemy to query tables
            with engine.connect() as connection:
                result = connection.execute("""
                SELECT 
                    table_name 
                FROM 
                    information_schema.tables 
                WHERE 
                    table_schema = 'public'
                ORDER BY 
                    table_name
                """)
                return {"tables": [row[0] for row in result]}
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    query = """
    SELECT 
        table_name 
    FROM 
        information_schema.tables 
    WHERE 
        table_schema = 'public'
    ORDER BY 
        table_name
    """
    
    results = execute_query(query)
    if not results:
        return {"tables": []}
    
    return {"tables": [row["table_name"] for row in results]}

@app.get("/table-schema/{table_name}")
def table_schema(table_name: str):
    # Validate table name to prevent SQL injection
    valid_tables = ["batteries", "battery_summaries", "battery_cycles_interpolated", "excluded_channels"]
    if table_name not in valid_tables:
        raise HTTPException(status_code=400, detail="Invalid table name")
    
    # Try SQLAlchemy first if connection is available
    if using_postgres:
        try:
            # Use SQLAlchemy to query schema
            with engine.connect() as connection:
                result = connection.execute(text("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable 
                FROM 
                    information_schema.columns 
                WHERE 
                    table_name = :table_name 
                ORDER BY 
                    ordinal_position
                """), {"table_name": table_name})
                return {"columns": [dict(row) for row in result]}
        except Exception as e:
            print(f"SQLAlchemy query failed: {e}")
            print("Falling back to direct SQL query")
    
    # Fall back to direct SQL query
    query = """
    SELECT 
        column_name, 
        data_type, 
        is_nullable 
    FROM 
        information_schema.columns 
    WHERE 
        table_name = %s 
    ORDER BY 
        ordinal_position
    """
    
    results = execute_query(query, [table_name])
    if not results:
        return {"columns": []}
    
    return {"columns": results}

# Health check and connection test endpoints
@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/connection-test")
def connection_test():
    # Test SQLAlchemy connection
    sqlalchemy_status = "ok" if using_postgres else "failed"
    
    # Test direct PostgreSQL connection
    conn = get_db_connection()
    direct_status = "ok" if conn else "failed"
    if conn:
        conn.close()
    
    # Get database info if connection is successful
    db_info = {}
    if conn:
        try:
            db_info = execute_query("SELECT version(), current_database(), current_user")[0]
        except Exception as e:
            db_info = {"error": str(e)}
    
    return {
        "sqlalchemy_connection": sqlalchemy_status,
        "direct_connection": direct_status,
        "database_info": db_info,
        "connection_string": f"postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        "timestamp": datetime.now().isoformat()
    }

# Create tables if they don't exist
def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        
    # Test direct PostgreSQL connection
    conn = get_db_connection()
    if conn:
        print("Direct PostgreSQL connection successful")
        
        # Check if tables exist
        tables = execute_query("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        print(f"Found {len(tables)} tables in database")
        for table in tables:
            print(f"  - {table['table_name']}")
            
        conn.close()

@app.on_event("startup")
async def startup_event():
    print("Starting up FastAPI application...")
    init_db()
    print("FastAPI application started successfully")

if __name__ == "__main__":
    import uvicorn
    print("Starting Solar Battery API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
