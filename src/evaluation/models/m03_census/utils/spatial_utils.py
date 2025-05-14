from typing import Dict, Tuple, List, Optional, Any
from math import radians, sin, cos, asin, sqrt
from collections import defaultdict
import requests
import os

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two points on Earth.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        Distance in kilometers
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # km

def calculate_distance_to_affected_area(agent_lat: float, agent_lon: float, cells: Dict[str, Any]) -> float:
    """Calculate the minimum distance from an agent to any affected cell in the proposal.
    
    Args:
        agent_lat: Agent's latitude
        agent_lon: Agent's longitude
        cells: Dictionary of cells from the proposal
        
    Returns:
        Minimum distance in kilometers
    """
    min_distance = float('inf')
    for cell in cells.values():
        bbox = cell.get('bbox', {})
        if not bbox:
            continue
        cell_lat = (bbox['north'] + bbox['south']) / 2
        cell_lon = (bbox['east'] + bbox['west']) / 2
        distance = calculate_haversine_distance(agent_lat, agent_lon, cell_lat, cell_lon)
        min_distance = min(min_distance, distance)
    print(f"DEBUG: Agent {agent_lat}, {agent_lon} is {min_distance:.2f}km from affected area")
    return min_distance

def get_neighborhood_name(lat: float, lon: float) -> Optional[str]:
    """Get neighborhood name from coordinates using Google Maps API."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return None
    params = {
        'latlng': f'{lat},{lon}',
        'result_type': 'neighborhood',
        'key': api_key
    }
    try:
        response = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params).json()
        for result in response.get('results', []):
            for component in result.get('address_components', []):
                if 'neighborhood' in component.get('types', []):
                    return component['long_name']
    except Exception as e:
        print(f"Warning: Failed to get neighborhood name: {str(e)}")
    return None

def get_nearby_transit(coordinates_by_zone: Dict[str, List[Tuple[float, float]]]) -> Optional[str]:
    """Get nearby transit information using Google Places API."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return None
    all_coords = [coord for coords in coordinates_by_zone.values() for coord in coords]
    if not all_coords:
        return None
    central_lat = sum(c[0] for c in all_coords) / len(all_coords)
    central_lon = sum(c[1] for c in all_coords) / len(all_coords)
    params = {
        'location': f'{central_lat},{central_lon}',
        'radius': 800,  # 800 meters ~ 0.5 miles
        'type': 'transit_station',
        'key': api_key
    }
    try:
        response = requests.get("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params=params).json()
        stations = [place['name'] for place in response.get('results', [])[:3]]
        if stations:
            return f"The area is served by public transit including {', '.join(stations)}. "
    except Exception as e:
        print(f"Warning: Failed to get transit information: {str(e)}")
    return None

def create_proposal_description(proposal: Dict[str, Any]) -> str:
    """Generate a richer, geo-aware description for a rezoning proposal."""
    height_limits = proposal.get("heightLimits", {})
    default_height = height_limits.get("default", 0)
    grid_config = proposal.get("gridConfig", {})
    cell_size = grid_config.get("cellSize", 100)
    cells = proposal.get("cells", {})

    zone_info = defaultdict(lambda: {'cells': [], 'coordinates': [], 'count': 0})

    for cell_id, cell in cells.items():
        try:
            category = cell.get("category", "unknown")
            height = cell.get("heightLimit", default_height)
            zone_type = (category, height)

            bbox = cell.get("bbox", {})
            if bbox:
                lat_center = (bbox["north"] + bbox["south"]) / 2
                lon_center = (bbox["east"] + bbox["west"]) / 2
                zone_info[zone_type]['cells'].append(cell)
                zone_info[zone_type]['coordinates'].append((lat_center, lon_center))
                zone_info[zone_type]['count'] += 1
        except (KeyError, TypeError) as e:
            print(f"Warning: Could not process cell {cell_id}: {str(e)}")
            continue

    desc = f"This rezoning proposal affects {len(cells)} city blocks in San Francisco"
    if len(cells) > 0:
        desc += f", each approximately {cell_size} meters square"
    desc += ". "

    zone_descriptions = []
    for (category, height), info in zone_info.items():
        try:
            num_blocks = info['count']
            if num_blocks == 0:
                continue

            area_sqm = num_blocks * (cell_size * cell_size)
            area_acres = area_sqm * 0.000247105

            coords = info['coordinates']
            if coords:
                central_lat = sum(c[0] for c in coords) / len(coords)
                central_lon = sum(c[1] for c in coords) / len(coords)
                neighborhood = get_neighborhood_name(central_lat, central_lon)

                zone_desc = (
                    f"{num_blocks} blocks ({area_acres:.1f} acres) zoned for "
                    f"{category.replace('_', ' ')} with a height limit of {height} feet"
                )
                if neighborhood:
                    zone_desc += f" in the {neighborhood} area"
                zone_descriptions.append(zone_desc)
        except Exception as e:
            print(f"Warning: Could not process zone {category}_{height}: {str(e)}")
            continue

    if zone_descriptions:
        desc += "The proposal includes: " + "; ".join(zone_descriptions[:3])
        if len(zone_descriptions) > 3:
            desc += f"; and {len(zone_descriptions) - 3} additional zoning changes"
        desc += ". "

    desc += (
        "This rezoning would affect local housing capacity, neighborhood character, "
        "and urban development patterns. The height limits are designed to balance housing needs "
        "with neighborhood context. "
    )

    transit_info = get_nearby_transit({
        f"{cat}_{h}": info['coordinates']
        for (cat, h), info in zone_info.items()
        if info['coordinates']
    })
    if transit_info:
        desc += transit_info

    print(f"DEBUG: Final proposal description:\n{desc.strip()}")
    return desc.strip()
