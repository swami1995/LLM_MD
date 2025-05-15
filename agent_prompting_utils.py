import json
import os

def save_profiles(n, m, folder_name, profile_generator):
    """
    Generates and saves 'n' agent profiles and 'm' user profiles,
    each group to a *single* JSON file within the specified folder.

    Args:
        n: The number of agent profiles to generate.
        m: The number of user profiles to generate.
        folder_name: The name of the folder to save the files in.
        profile_generator: An instance of the ProfileGenerator class.
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created directory: {folder_name}")

    agent_profiles = []
    for i in range(1, n + 1):
        agent_profile, agent_profile_refined, _ = profile_generator.generate_and_validate_agent()
        if agent_profile_refined:
            profile_to_save = agent_profile_refined
        else:
            profile_to_save = agent_profile

        if profile_to_save: # make sure profile exists
            agent_profiles.append(profile_to_save)
            print(f"Generated agent profile {i}")
        else: # error handling
            print(f"Could not generate a valid agent profile for iteration {i}.")

    agent_filename = os.path.join(folder_name, "agent_profiles_fixed.json")
    try:
        with open(agent_filename, 'w') as f:
            json.dump(agent_profiles, f, indent=4)  # Save the entire list
        print(f"Saved all agent profiles to {agent_filename}")
    except Exception as e:
        print(f"Error saving agent profiles to {agent_filename}: {e}")


    user_profiles = []
    for i in range(1, m + 1):
        user_profile, user_profile_refined, _ = profile_generator.generate_and_validate_user()
        if user_profile_refined:
            profile_to_save = user_profile_refined
        else:
            profile_to_save = user_profile

        if profile_to_save: # make sure profile exists
            user_profiles.append(profile_to_save)
            print(f"Generated user profile {i}")
        else: # error handling
            print(f"Could not generate a valid user profile for iteration {i}.")


    user_filename = os.path.join(folder_name, "user_profiles_fixed.json")
    try:
        with open(user_filename, 'w') as f:
            json.dump(user_profiles, f, indent=4)  # Save the entire list
        print(f"Saved all user profiles to {user_filename}")
    except Exception as e:
        print(f"Error saving user profiles to {user_filename}: {e}")


def load_profiles(folder_name):
    """
    Loads agent and user profiles from their respective single JSON files
    within the specified folder.

    Args:
        folder_name: The name of the folder containing the files.

    Returns:
        A tuple: (agent_profiles, user_profiles), where each element is a
        list of profile dictionaries.  Returns (None, None) if an error occurs
        or if either file is not found.
    """

    agent_filename = os.path.join(folder_name, "agent_profiles_fixed.json")
    user_filename = os.path.join(folder_name, "user_profiles_fixed.json")

    agent_profiles = None
    user_profiles = None

    try:
        with open(agent_filename, 'r') as f:
            agent_profiles = json.load(f)
        print(f"Successfully loaded agent profiles from {agent_filename}")
    except FileNotFoundError:
        print(f"Agent profiles file not found: {agent_filename}")
    except Exception as e:
        print(f"Error loading agent profiles from {agent_filename}: {e}")

    try:
        with open(user_filename, 'r') as f:
            user_profiles = json.load(f)
        print(f"Successfully loaded user profiles from {user_filename}")
    except FileNotFoundError:
        print(f"User profiles file not found: {user_filename}")
    except Exception as e:
        print(f"Error loading user profiles from {user_filename}: {e}")


    return agent_profiles, user_profiles


def save_constraints(constraints, filename):
    """
    Saves a constraints dictionary to a JSON file.

    Args:
        constraints: The dictionary to save.
        filename: The name of the file to save to (e.g., "agent_constraints.json").
    """
    filepath = os.path.join(filename) # use filepath.
    try:
        with open(filepath, 'w') as f:
            json.dump(constraints, f, indent=4)
        print(f"Successfully saved constraints to {filepath}")
    except Exception as e:
        print(f"Error saving constraints to {filepath}: {e}")

def load_constraints(filename):
    """
    Loads a constraints dictionary from a JSON file.

    Args:
        filename: The name of the file to load from (e.g., "agent_constraints.json").

    Returns:
        The loaded dictionary, or None if an error occurred.
    """
    filepath = os.path.join(filename) # use filepath.
    try:
        with open(filepath, 'r') as f:
            constraints = json.load(f)
        print(f"Successfully loaded constraints from {filepath}")
        return constraints
    except FileNotFoundError:
        print(f"Constraints file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading constraints from {filepath}: {e}")
        return None

def load_prompts(filename):
    """
    Loads conversational prompts from a JSON file.

    Args:
        filename: The name of the file to load from (e.g., "generated_prompts.json").

    Returns:
        The loaded list of prompts, or None if an error occurred.
    """
    filepath = os.path.join(filename) # use filepath.
    try:
        with open(filepath, 'r') as f:
            prompts = json.load(f)
        print(f"Successfully loaded prompts from {filepath}")
        return prompts
    except FileNotFoundError:
        print(f"Prompts file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading prompts from {filepath}: {e}")
        return None