"""
Shared test fixtures for the project.
"""

import os
import sys
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def project_root():
    return os.path.join(os.path.dirname(__file__), '..')


@pytest.fixture(scope="session")
def db_path(project_root):
    path = os.path.join(project_root, 'ecom.db')
    if not os.path.exists(path):
        pytest.skip("Database not generated yet")
    return path
