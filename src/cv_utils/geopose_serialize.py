import json
from dataclasses import asdict

from cv_utils.geopose_format import *

# Example: Create a GeoPoseReq instance
pose = GeoPoseReq(
    id="123",
    timestamp=1680000000000,
    type="geopose",
    sensors=[
        Sensor(
            id="sensor1",
            type="camera",
            name="Camera 1"
        )
    ],
    sensorReadings=SensorReadings(),
    priorPoses=None
)

# Convert to a dictionary (recursively)
data_dict = asdict(pose)

# Save to a JSON file
with open("geopose_request.json", "w") as f:
    json.dump(data_dict, f, indent=2)

print("JSON file created!")
