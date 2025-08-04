"""Base task class"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """Abstract base class for all tasks"""
    
    def __init__(self, name: str):
        """Initialize base task
        
        Args:
            name: Task name
        """
        self.name = name
    
    @abstractmethod
    def execute(self):
        """Execute the task - must be implemented by subclasses"""
        pass
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"