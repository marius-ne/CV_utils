from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CameraParam:
    model: Optional[str] = None
    modelParams: Optional[List[float]] = None
    minMaxDepth: Optional[List[float]] = None
    minMaxDisparity: Optional[List[float]] = None


@dataclass
class Privacy:
    dataRetention: List[str]
    dataAcceptableUse: List[str]
    dataSanitizationApplied: List[str]
    dataSanitizationRequested: List[str]


@dataclass
class ImageOrientation:
    mirrored: bool
    rotation: int


@dataclass
class CameraReading:
    timestamp: int
    sensorId: str
    privacy: Privacy
    sequenceNumber: int
    imageFormat: str
    size: List[int]
    imageBytes: str
    imageOrientation: Optional[ImageOrientation] = None
    params: Optional[CameraParam] = None


@dataclass
class GeolocationReading:
    timestamp: int
    sensorId: str
    privacy: Privacy
    latitude: float
    longitude: float
    altitude: float
    accuracy: float
    altitudeAccuracy: float
    heading: float
    speed: float


@dataclass
class WiFiReading:
    timestamp: int
    sensorId: str
    privacy: Privacy
    BSSID: str
    frequency: int
    RSSI: int
    SSID: str
    scanTimeStart: int
    scanTimeEnd: int


@dataclass
class BluetoothReading:
    timestamp: int
    sensorId: str
    privacy: Privacy
    address: str
    RSSI: int
    name: str


@dataclass
class AccelerometerReading:
    timestamp: int
    sensorId: str
    privacy: Privacy
    x: float
    y: float
    z: float


@dataclass
class GyroscopeReading:
    timestamp: int
    sensorId: str
    privacy: Privacy
    x: float
    y: float
    z: float


@dataclass
class MagnetometerReading:
    timestamp: int
    sensorId: str
    privacy: Privacy
    x: float
    y: float
    z: float


@dataclass
class Position:
    lon: float
    lat: float
    h: float


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class Sensor:
    id: str
    type: str
    name: Optional[str] = None
    model: Optional[str] = None
    rigIdentifier: Optional[str] = None
    rigRotation: Optional[Quaternion] = None
    rigTranslation: Optional[Vector3] = None


@dataclass
class SensorReadings:
    cameraReadings: Optional[List[CameraReading]] = None
    geolocationReadings: Optional[List[GeolocationReading]] = None
    wiFiReadings: Optional[List[WiFiReading]] = None
    bluetoothReadings: Optional[List[BluetoothReading]] = None
    accelerometerReadings: Optional[List[AccelerometerReading]] = None
    gyroscopeReadings: Optional[List[GyroscopeReading]] = None
    magnetometerReadings: Optional[List[MagnetometerReading]] = None


@dataclass
class GeoPoseAccuracy:
    position: float
    orientation: float


@dataclass
class GeoPose:
    position: Position
    quaternion: Quaternion


@dataclass
class GeoPoseResp:
    id: str
    timestamp: int
    accuracy: GeoPoseAccuracy
    type: str
    geopose: GeoPose


@dataclass
class GeoPoseReq:
    id: str
    timestamp: int
    type: str
    sensors: List[Sensor]
    sensorReadings: SensorReadings
    priorPoses: Optional[List[GeoPoseResp]] = None
