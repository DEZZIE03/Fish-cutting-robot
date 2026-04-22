import pyzed.sl as sl

print("Creating camera object...")
zed = sl.Camera()

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.camera_fps = 30

status = zed.open(init)

print("Open status:", status)

if status != sl.ERROR_CODE.SUCCESS:
    print("ZED open failed")
    exit(1)

print("ZED opened successfully")
zed.close()
print("ZED closed successfully")
