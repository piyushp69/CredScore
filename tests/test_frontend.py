import pytest
from streamlit.testing.v1 import AppTest
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_dashboard_loads():
    # Use AppTest to run the Streamlit app
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dashboard.py")
    at = AppTest.from_file(app_path)
    at.run()
    
    assert not at.exception
    assert "Applicant Underwriting" in [t.value for t in at.title]
    
def test_dashboard_form_submit(mocker):
    # Mock the requests.post call so we don't hit the real backend
    mock_post = mocker.patch("requests.post")
    
    class MockResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "probability_of_default": 0.45,
                "top_3_positive_contributions": {"FEATURE_1": 0.1, "FEATURE_2": 0.05, "FEATURE_3": 0.02},
                "top_3_negative_contributions": {"FEATURE_4": -0.1, "FEATURE_5": -0.05, "FEATURE_6": -0.02}
            }
            
    mock_post.return_value = MockResponse()
    
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dashboard.py")
    at = AppTest.from_file(app_path).run()
    
    assert not at.exception
    
    # Fill out the form
    at.number_input[0].set_value(0.6)
    
    # Submit the form
    at.button[0].click().run()
    
    assert not at.exception
    
    # Check that mock was called
    mock_post.assert_called_once()
    
    # Verify that gauge chart is rendered in the UI
    # Subheader for decision should be present
    assert "Underwriting Decision" in [sh.value for sh in at.subheader]
