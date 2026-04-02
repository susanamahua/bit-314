import ee
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_gee():
    """Initializes Google Earth Engine."""
    project_id = os.environ.get('EE_PROJECT_ID')
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        logger.info("Earth Engine initialized successfully.")
    except ee.EEException as e:
        logger.warning(f"Earth Engine initialization failed: {e}. Checking authentication...")
        try:
            # If initialization fails, it usually means authentication is required.
            logger.info("Please run `earthengine authenticate` in your terminal if this fails.")
            ee.Authenticate()
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            logger.info("Earth Engine authenticated and initialized successfully.")
        except Exception as auth_e:
            logger.error(f"Earth Engine authentication failed: {auth_e}. Please configure your GCP project.")
