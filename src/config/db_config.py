# Database configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'Rayana3',
        'USER': 'postgres',
        'PASSWORD': 'Rayana2024', 
        'HOST': 'localhost',
        'PORT': '5432',
    },
    'OPTIONS': {
        'client_encoding': 'UTF8',
    }
}

# Connection string for SQLAlchemy
DATABASE_URL = f"postgresql://{DATABASES['default']['USER']}:{DATABASES['default']['PASSWORD']}@{DATABASES['default']['HOST']}:{DATABASES['default']['PORT']}/{DATABASES['default']['NAME']}"