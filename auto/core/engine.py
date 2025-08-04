"""Main automation engine"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class AutomationEngine:
    """Main automation engine for running tasks"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the automation engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.tasks = []
        logger.info("AutomationEngine initialized")
    
    def add_task(self, task):
        """Add a task to the engine
        
        Args:
            task: Task object to add
        """
        self.tasks.append(task)
        logger.debug(f"Added task: {task}")
    
    def run(self):
        """Execute all registered tasks"""
        logger.info(f"Running {len(self.tasks)} tasks")
        for task in self.tasks:
            try:
                task.execute()
            except Exception as e:
                logger.error(f"Task failed: {e}")
    
    def clear_tasks(self):
        """Clear all registered tasks"""
        self.tasks = []
        logger.debug("Cleared all tasks")