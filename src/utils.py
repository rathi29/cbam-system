import random
from datetime import datetime, timedelta
import uuid


def generate_mock_shipments(count=15):
    materials = ['Steel', 'Cement', 'Aluminum', 'Fertilizer', 'Chemicals']
    countries = ['China', 'India', 'Russia', 'Turkey', 'Ukraine']
    
    shipments = []
    for i in range(count):
        shipment = {
            'shipment_id': f'SHP-{uuid.uuid4().hex[:8].upper()}',  # Unique ID
            'material_type': random.choice(materials),
            'tonnage': round(random.uniform(50, 400), 2),
            'country_of_origin': random.choice(countries),
            'direct_emissions': round(random.uniform(0.5, 8.0), 2),
            'audit_flag': 'PASS' if random.random() > 0.2 else 'FLAG_MISSING_EMISSIONS',
            'created_at': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            'source_document': f'manifest_{i}.pdf'
        }
        shipments.append(shipment)
    return shipments

def validate_shipment(data):
    required = ['shipment_id', 'material_type', 'tonnage', 'country_of_origin', 'direct_emissions']
    errors = []
    
    for field in required:
        if field not in data or data[field] is None:
            errors.append(f"Missing: {field}")
    
    if data.get('tonnage', 0) <= 0:
        errors.append("Tonnage must be > 0")
    if data.get('direct_emissions', 0) < 0:
        errors.append("Emissions cannot be negative")
    
    return len(errors) == 0, errors