

def create_user_representatives(user_profiles, num_representatives=3):
    """
    Cluster users and create representatives for each segment.
    
    Parameters:
    - user_profiles: List of user profiles
    - num_representatives: Target number of representatives to create
    
    Returns:
    - List of representative definitions
    """
    # For simplicity, we'll use a predefined segmentation based on technical proficiency
    segments = {
        "technical": [],   # High technical proficiency
        "non_technical": [], # Low technical proficiency
        "balanced": []     # Medium technical proficiency
    }
    
    # Assign users to segments based on technical proficiency
    for user_id, profile in enumerate(user_profiles):
        tech_level = profile.get("technical_proficiency", "Medium")
        
        if tech_level == "High":
            segments["technical"].append(user_id)
        elif tech_level == "Low":
            segments["non_technical"].append(user_id)
        else:
            segments["balanced"].append(user_id)
    
    # Create representative profiles
    representatives = []
    
    for segment_name, user_ids in segments.items():
        if user_ids:  # Only create representatives for segments with users
            # Create a representative profile based on segment
            if segment_name == "technical":
                rep_profile = {
                    "name": "Technical User Representative",
                    "description": "Represents users with high technical proficiency",
                    "focus": ["Factual_Correctness", "Technical_Detail", "Efficiency"],
                    "value_weights": {
                        "Factual_Correctness": 0.9,
                        "Process_Reliability": 0.8,
                        "Transparency": 0.7,
                        "Trust_Calibration": 0.8
                    }
                }
            elif segment_name == "non_technical":
                rep_profile = {
                    "name": "Non-Technical User Representative",
                    "description": "Represents users with low technical proficiency",
                    "focus": ["Helpfulness", "Clarity", "Patience"],
                    "value_weights": {
                        "Communication_Quality": 0.9,
                        "Problem_Resolution": 0.8,
                        "Value_Alignment": 0.7,
                    }
                }
            else:  # balanced
                rep_profile = {
                    "name": "Average User Representative",
                    "description": "Represents users with moderate technical proficiency",
                    "focus": ["Balance", "Effectiveness", "Usability"],
                    "value_weights": {
                        "Communication_Quality": 0.8,
                        "Problem_Resolution": 0.8,
                        "Value_Alignment": 0.7,
                        "Transparency": 0.7
                    }
                }
            
            representatives.append({
                "segment": segment_name,
                "profile": rep_profile,
                "user_ids": user_ids,
                "size": len(user_ids)
            })
    
    return representatives
