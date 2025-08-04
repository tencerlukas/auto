"""Tests for the automation engine"""

import unittest
from unittest.mock import Mock
from auto.core import AutomationEngine


class TestAutomationEngine(unittest.TestCase):
    """Test cases for AutomationEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = AutomationEngine()
    
    def test_init(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.config, {})
        self.assertEqual(self.engine.tasks, [])
    
    def test_add_task(self):
        """Test adding tasks"""
        mock_task = Mock()
        self.engine.add_task(mock_task)
        self.assertIn(mock_task, self.engine.tasks)
    
    def test_clear_tasks(self):
        """Test clearing tasks"""
        mock_task = Mock()
        self.engine.add_task(mock_task)
        self.engine.clear_tasks()
        self.assertEqual(self.engine.tasks, [])


if __name__ == '__main__':
    unittest.main()