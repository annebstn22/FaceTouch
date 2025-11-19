import json
import socket
import threading
import time
import random
from Cocoa import NSWindow, NSView, NSColor, NSScreen, NSTimer, NSWindowStyleMaskBorderless
from AppKit import NSFloatingWindowLevel
from PyObjCTools import AppHelper

# Storage for dots
dots = []  # Each dot: {"x": float, "y": float, "color": (r,g,b), "size": float, "timestamp": float}
DURATION = 300  # 5 minutes

# Colors (0..1)
FACE_COLORS = {
    "orange": (0xdd/255, 0x55/255, 0x18/255),
    "yellow": (0xde/255, 0xb9/255, 0x4f/255),
    "pink":   (0xbb/255, 0x30/255, 0x79/255),
    "red":    (0xcd/255, 0x37/255, 0x2a/255),
    "blue":   (0x0c/255, 0x56/255, 0xb2/255),
    "green":  (0x0c/255, 0x80/255, 0x4d/255),
}

class Overlay(NSWindow):
    def canBecomeKeyWindow(self):
        return False
    def canBecomeMainWindow(self):
        return False

class OverlayView(NSView):
    def drawRect_(self, rect):
        now = time.time()
        dead = []

        for d in dots:
            age = now - d["timestamp"]
            if age > DURATION:
                dead.append(d)
                continue

            fade = max(0, 1 - age / DURATION)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                d["color"][0], d["color"][1], d["color"][2], fade
            ).set()

            x, y = d["x"], d["y"]
            r = d["size"]/2
            from Quartz import NSBezierPath, NSMakeRect
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(x-r, y-r, d["size"], d["size"])
            ).fill()

        for d in dead:
            dots.remove(d)
            
    # Add refresh method
    def refresh_(self, _):
        self.setNeedsDisplay_(True)

# UDP listener thread
def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 5055))
    print("UDP listener running on 127.0.0.1:5055")
    while True:
        data, _ = sock.recvfrom(4096)
        try:
            msg = json.loads(data.decode())
            print("Received message:", msg)
            region = msg.get("region", "red")
            color = FACE_COLORS.get(region, (1,0,0))

            # Center of main screen
            screen = NSScreen.mainScreen().frame()
            x = random.uniform(0, screen.size.width)
            y = random.uniform(0, screen.size.height)

            size = random.choice([20,30,45])

            dots.append({
                "x": x,
                "y": y,
                "color": color,
                "size": size,
                "timestamp": time.time()
            })
        except Exception as e:
            print("UDP parse error:", e)

def main():
    screen = NSScreen.mainScreen().frame()

    # Create window
    window = Overlay.alloc().initWithContentRect_styleMask_backing_defer_(
        screen,
        NSWindowStyleMaskBorderless,
        2,  # NSBackingStoreBuffered
        False
    )
    window.setLevel_(NSFloatingWindowLevel)
    window.setOpaque_(False)
    window.setBackgroundColor_(NSColor.clearColor())
    window.setIgnoresMouseEvents_(True)
    window.makeKeyAndOrderFront_(None)

    # Create view
    view = OverlayView.alloc().initWithFrame_(screen)
    window.setContentView_(view)

    # Add a test dot
    dots.append({
        "x": screen.size.width / 2,
        "y": screen.size.height / 2,
        "color": (1, 0, 0),  # red
        "size": 50,
        "timestamp": time.time()
    })

    # Refresh view every 30ms
    def refresh(_):
        view.setNeedsDisplay_(True)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.03, view, 'refresh:', None, True
    )
    # Bind the selector
    setattr(view, "refresh:", refresh)

    # Start UDP listener
    threading.Thread(target=udp_listener, daemon=True).start()

    # Start event loop
    AppHelper.runEventLoop()

if __name__ == "__main__":
    main()
