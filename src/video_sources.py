"""
SKYWATCH - Test Video Kaynaklari

Canli kamera yokken izleme ekranini video dosyalariyla test etmek icin
bu dosyadaki yollari guncelleyin.
"""

# Kamera ID -> video dosya yolu
VIDEO_SOURCES: dict[str, str] = {
    "CAM_01": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera1.mp4",
    "CAM_02": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera2.webm",
    "CAM_03": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera3.mp4",
    "CAM_04": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera4.mp4",
    "CAM_05": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera5.mp4",
    "CAM_06": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera6.mp4",
    "CAM_07": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera7.mp4",
    "CAM_08": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera8.mp4",
    "CAM_09": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera9.mp4",
    "CAM_10": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera10.mp4",
    "CAM_11": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera11.mp4",
    "CAM_12": r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera12.mp4",
}

# Varsayilan aktif test kaynaklari (2 kamera stabil test modu)
ACTIVE_VIDEO_SOURCE_IDS = ["CAM_01", "CAM_03"]

def get_active_video_sources(max_active_cameras: int = 2) -> dict[str, str]:
    """Sadece aktif test kaynaklarini döndürür."""
    active_ids = ACTIVE_VIDEO_SOURCE_IDS[:max_active_cameras]
    return {cid: VIDEO_SOURCES[cid] for cid in active_ids if cid in VIDEO_SOURCES}

# Kamera ID -> UI'da gorunecek isim
CAMERA_LABELS: dict[str, str] = {
    "CAM_01": "Kamera 1",
    "CAM_02": "Kamera 2",
    "CAM_03": "Kamera 3",
    "CAM_04": "Kamera 4",
    "CAM_05": "Kamera 5",
    "CAM_06": "Kamera 6",
    "CAM_07": "Kamera 7",
    "CAM_08": "Kamera 8",
    "CAM_09": "Kamera 9",
    "CAM_10": "Kamera 10",
    "CAM_11": "Kamera 11",
    "CAM_12": "Kamera 12",
}