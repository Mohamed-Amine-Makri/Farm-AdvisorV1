from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from src.database.models import Base, Farmer, Farm, Recommendation, Plan, Conversation, Message
from src.config.db_config import DATABASE_URL
import uuid
from typing import Optional  # Add this import

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionLocal)

class Repository:
    def __init__(self):
        self.session = db_session()
    
    def close(self):
        self.session.close()
    
    # Farmer operations
    def create_farmer(self, name=None, phone_number=None, email=None):
        farmer = Farmer(
            name=name,
            phone_number=phone_number,
            email=email
        )
        self.session.add(farmer)
        self.session.commit()
        return farmer
    
    def get_farmer_by_id(self, farmer_id):
        return self.session.query(Farmer).filter(Farmer.id == farmer_id).first()
    
    # Farm operations
    def create_farm(self, farmer_id, location, surface_area, soil_type=None, current_plants=None, weather_conditions=None):
        farm = Farm(
            farmer_id=farmer_id,
            location=location,
            surface_area=surface_area,
            soil_type=soil_type,
            current_plants=current_plants,
            weather_conditions=weather_conditions
        )
        self.session.add(farm)
        self.session.commit()
        return farm
    
    def update_farm(self, farm_id, **kwargs):
        farm = self.session.query(Farm).filter(Farm.id == farm_id).first()
        if farm:
            for key, value in kwargs.items():
                setattr(farm, key, value)
            self.session.commit()
        return farm
    
    def get_farm_by_id(self, farm_id):
        return self.session.query(Farm).filter(Farm.id == farm_id).first()
    
    def get_farms_by_farmer_id(self, farmer_id):
        return self.session.query(Farm).filter(Farm.farmer_id == farmer_id).all()
    
    def get_latest_farm(self):
        """Get the most recently created farm"""
        return self.session.query(Farm).order_by(Farm.created_at.desc()).first()
    
    # Recommendation operations
    def create_recommendation(self, farm_id, recommended_plants, irrigation_methods, reasoning=None):
        recommendation = Recommendation(
            farm_id=farm_id,
            recommended_plants=recommended_plants,
            irrigation_methods=irrigation_methods,
            reasoning=reasoning
        )
        self.session.add(recommendation)
        self.session.commit()
        return recommendation
    
    def get_recommendations_by_farm_id(self, farm_id):
        return self.session.query(Recommendation).filter(Recommendation.farm_id == farm_id).all()
    
    # Plan operations
    def create_plan(self, farm_id: int, monthly_schedule: str, 
                    planting_schedule: Optional[str] = None,
                    irrigation_schedule: Optional[str] = None, 
                    soil_preparation: Optional[str] = None,
                    harvest_times: Optional[str] = None,
                    seasonal_considerations: Optional[str] = None) -> Plan:  # Changed from models.Plan to Plan
        """Create a new agricultural plan for a farm"""
        plan = Plan(  # Use Plan directly instead of models.Plan
            farm_id=farm_id,
            monthly_schedule=monthly_schedule,
            planting_schedule=planting_schedule,
            irrigation_schedule=irrigation_schedule,
            soil_preparation=soil_preparation,
            harvest_times=harvest_times,
            seasonal_considerations=seasonal_considerations
        )
        self.session.add(plan)
        self.session.commit()
        return plan
    
    def get_plans_by_farm_id(self, farm_id):
        return self.session.query(Plan).filter(Plan.farm_id == farm_id).all()
    
    # Conversation operations
    def create_conversation(self, farmer_id=None):
        thread_id = str(uuid.uuid4())
        conversation = Conversation(
            farmer_id=farmer_id,
            thread_id=thread_id
        )
        self.session.add(conversation)
        self.session.commit()
        return conversation
    
    def get_conversation_by_thread_id(self, thread_id):
        return self.session.query(Conversation).filter(Conversation.thread_id == thread_id).first()
    
    # Message operations
    def add_message(self, conversation_id, role, content):
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )
        self.session.add(message)
        self.session.commit()
        return message
    
    def get_messages_by_conversation_id(self, conversation_id):
        return self.session.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()

    # Helper methods for agent operations
    def save_farm_data_from_extraction(self, farmer_id, location, surface_area, soil_type=None, current_plants=None, weather_conditions=None):
        # Check if the farmer has an existing farm
        existing_farms = self.get_farms_by_farmer_id(farmer_id)
        
        if existing_farms:
            # Update the most recent farm
            farm = existing_farms[-1]
            return self.update_farm(
                farm.id,
                location=location,
                surface_area=surface_area,
                soil_type=soil_type,
                current_plants=current_plants,
                weather_conditions=weather_conditions
            )
        else:
            # Create a new farm
            return self.create_farm(
                farmer_id=farmer_id,
                location=location,
                surface_area=surface_area,
                soil_type=soil_type,
                current_plants=current_plants,
                weather_conditions=weather_conditions
            )
    
    def save_conversation_history(self, thread_id, user_message, assistant_message, farmer_id=None):
        # Get or create conversation
        conversation = self.get_conversation_by_thread_id(thread_id)
        if not conversation:
            conversation = self.create_conversation(farmer_id)
        
        # Add messages
        self.add_message(conversation.id, 'user', user_message)
        self.add_message(conversation.id, 'assistant', assistant_message)
        
        return conversation