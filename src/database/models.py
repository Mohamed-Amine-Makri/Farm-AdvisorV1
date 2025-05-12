from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import inspect
import datetime
from src.config.db_config import DATABASE_URL
from sqlalchemy import text
Base = declarative_base()

class Farmer(Base):
    __tablename__ = 'farmers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=True)
    phone_number = Column(String(20), nullable=True)
    email = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    farms = relationship("Farm", back_populates="farmer")
    conversations = relationship("Conversation", back_populates="farmer")

class Farm(Base):
    __tablename__ = 'farms'
    
    id = Column(Integer, primary_key=True)
    farmer_id = Column(Integer, ForeignKey('farmers.id'))
    location = Column(String(100), nullable=False)  # Location in Tunisia
    surface_area = Column(Float, nullable=False)  # in hectares
    soil_type = Column(String(50), nullable=True)
    current_plants = Column(String(255), nullable=True)
    weather_conditions = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    farmer = relationship("Farmer", back_populates="farms")
    recommendations = relationship("Recommendation", back_populates="farm")
    plans = relationship("Plan", back_populates="farm")
    plants = relationship("Plant", back_populates="farm")

class Plant(Base):
    __tablename__ = 'plants'
    
    id = Column(Integer, primary_key=True)
    farm_id = Column(Integer, ForeignKey('farms.id'))
    name = Column(String(100), nullable=False)
    variety = Column(String(100), nullable=True)
    planting_date = Column(DateTime, nullable=True)
    harvest_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    farm = relationship("Farm", back_populates="plants")

class Recommendation(Base):
    __tablename__ = 'recommendations'
    
    id = Column(Integer, primary_key=True)
    farm_id = Column(Integer, ForeignKey('farms.id'))
    recommended_plants = Column(String(255), nullable=False)
    irrigation_methods = Column(String(255), nullable=False)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    farm = relationship("Farm", back_populates="recommendations")

class Plan(Base):
    __tablename__ = 'plans'
    
    id = Column(Integer, primary_key=True)
    farm_id = Column(Integer, ForeignKey('farms.id'))
    monthly_schedule = Column(Text, nullable=False)  # JSON string of monthly activities
    planting_schedule = Column(Text, nullable=True)
    irrigation_schedule = Column(Text, nullable=True)
    soil_preparation = Column(Text, nullable=True)
    harvest_times = Column(Text, nullable=True)
    seasonal_considerations = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    farm = relationship("Farm", back_populates="plans")

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    farmer_id = Column(Integer, ForeignKey('farmers.id'))
    thread_id = Column(String(100), nullable=False, unique=True)  # For maintaining conversation context
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    farmer = relationship("Farmer", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

# Initialize the database
def init_db():
    """Initialize database with tables created in the correct order"""
    engine = create_engine(DATABASE_URL)
    
    try:
        # Connect to the database
        with engine.connect() as connection:
            # Drop tables with CASCADE using raw SQL to handle dependencies
            connection.execute(text("""
                DROP TABLE IF EXISTS messages CASCADE;
                DROP TABLE IF EXISTS conversations CASCADE;
                DROP TABLE IF EXISTS recommendations CASCADE;
                DROP TABLE IF EXISTS plans CASCADE;
                DROP TABLE IF EXISTS pump CASCADE;
                DROP TABLE IF EXISTS farm_web CASCADE;
                DROP TABLE IF EXISTS irrigation CASCADE;
                DROP TABLE IF EXISTS minifarms CASCADE;
                DROP TABLE IF EXISTS farms CASCADE;
                DROP TABLE IF EXISTS farmers CASCADE;
            """))
            connection.commit()
        print("Dropped existing tables")
        
        # Create tables in specific order
        tables_in_order = [
            Farmer.__table__,
            Farm.__table__,
            Conversation.__table__,
            Plant.__table__,  # Add this line
            Recommendation.__table__,
            Plan.__table__,
            Message.__table__
        ]
        
        # Create each table in order
        for table in tables_in_order:
            table.create(engine, checkfirst=True)
            print(f"Created table: {table.name}")
        
        print("All tables created successfully")
        return engine
        
    except Exception as e:
        print(f"Error during database initialization: {str(e)}")
        raise