"""
Utility functions for PowerFactory operations.
"""


def init_project(app, project_path: str) -> bool:
    """
    Initialize a PowerFactory project.
    
    Args:
        app: PowerFactory application instance
        project_path: Full path to the project (e.g., "User\\ProjectName")
        
    Returns:
        True if successful, False otherwise
    """
    err = app.ActivateProject(project_path)
    return err == 0
