# pylint: disable=all

import psycopg2
import pytest
import requests


@pytest.fixture(scope="module")
def db_connection():
    """Fixture to set up the database connection."""
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="example",
        host="localhost",
        port="5432",
    )
    yield conn
    conn.close()


def test_db_connection(db_connection):
    """Test to check connection to PostgreSQL database."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT version();")
    db_version = cursor.fetchone()
    assert db_version is not None
    assert "PostgreSQL" in db_version[0]
    cursor.close()


def test_adminer_access():
    """Test to check access to Adminer."""
    response = requests.get("http://localhost:8080")
    assert response.status_code == 200


def test_grafana_access():
    """Test to check access to Grafana."""
    response = requests.get("http://localhost:3000")
    assert response.status_code == 200
