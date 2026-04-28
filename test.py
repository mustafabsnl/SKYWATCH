import cv2
import time
import threading
import psutil

CAMERA_SOURCES = [
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera1.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera2.webm",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera3.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera4.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera5.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera6.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera7.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera8.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera9.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera10.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera11.mp4",
r"C:\Users\musta\OneDrive\Desktop\SKYWATCH\Kameralar\Kamera12.mp4",
]


class CameraTest:
    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.frames = 0
        self.fps = 0
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)

        # Buffer birikimini azalt
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

        if not cap.isOpened():
            print(f"{self.name} acilamadi!")
            return

        last_time = time.time()
        last_frames = 0

        while self.running:
            ret, frame = cap.read()

            if not ret:
                # video biterse başa sar
                cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                continue

            self.frames += 1

            now=time.time()
            if now-last_time>=1:
                self.fps=self.frames-last_frames
                last_frames=self.frames
                last_time=now

        cap.release()


tests=[]

for i,src in enumerate(CAMERA_SOURCES):
    cam=CameraTest(src,f"CAM_{i+1}")
    tests.append(cam)

    t=threading.Thread(target=cam.run,daemon=True)
    t.start()

try:
    while True:
        print("\n--- SISTEM DURUMU ---")
        print(f"CPU: {psutil.cpu_percent()}%")
        print(f"RAM: {psutil.virtual_memory().percent}%")

        total_fps=0

        for cam in tests:
            print(f"{cam.name}: {cam.fps} FPS")
            total_fps+=cam.fps

        print("Toplam FPS:",total_fps)

        time.sleep(1)

except KeyboardInterrupt:
    for cam in tests:
        cam.running=False

    print("Test durduruldu.")